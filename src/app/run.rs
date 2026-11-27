//! Async application runner, intended to run in a tokio runtime.

use eyre::WrapErr;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::signal::unix::{SignalKind, signal};
use tokio::{net::TcpListener, task::JoinSet, time::timeout};

use crate::network::{
    router::make_router,
    udp::{NetworkActivity, packet_handler_task, packet_monitor_task},
};
use crate::state::AppConfig;
use crate::state::AppState;

/// Number of seconds to wait at shutdown before killing threads
const SHUTDOWN_TIMEOUT_SECONDS: u64 = 5;

/// Spawns all program tasks, including:
/// - rfi solar enable/disable task
/// - bad input metric computation task
/// - packet recv task
/// - http server
///
/// # Errors
/// - Errors if either the http or UDP recv address cannot be bound.
/// - Errors if any task fails
pub async fn run(
    http_addr: SocketAddr,
    udp_addr: SocketAddr,
    config: Option<AppConfig>,
    receiver_threads: usize,
) -> eyre::Result<()> {
    let state = AppState::default();
    let mut errors: Vec<eyre::Report> = Vec::new();

    // spawn UDP receiver tasks
    let mut rx_tasks = spawn_rx_tasks(udp_addr, receiver_threads, &state);
    // spawn internal processing tasks
    let mut tasks = spawn_proc_tasks(config, &state);
    // spawn the HTTP server
    let mut server_tasks = spawn_http_tasks(http_addr, state).await?;

    // register shutdown signals
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    tokio::select! {
        _ = sigterm.recv() => tracing::warn!("received SIGTERM; shutting down..."),
        _ = sigint.recv() => tracing::warn!("received SIGINT; shutting down..."),
        Some(result) = rx_tasks.join_next() => {
            errors.push(eyre::eyre!("a network task exited early: {result:?}"));
        },
        Some(result) = tasks.join_next() => {
            errors.push(eyre::eyre!("a processing task exited early: {result:?}"));
        },
        Some(result) = server_tasks.join_next() => {
            errors.push(eyre::eyre!("a server task exited early: {result:?}"));
        },
    };

    // shut down everything cleanly
    let rx_drain = async {
        rx_tasks.abort_all();
        while rx_tasks.join_next().await.is_some() {}
        tracing::info!("shut down receivers");
    };
    let server_drain = async {
        server_tasks.abort_all();
        while server_tasks.join_next().await.is_some() {}
        tracing::info!("shut down HTTP server");
    };
    let task_drain = async {
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        tracing::info!("shut down processing tasks");
    };
    // exit if joining threads timed out
    if timeout(Duration::from_secs(SHUTDOWN_TIMEOUT_SECONDS), rx_drain)
        .await
        .is_err()
    {
        errors.push(eyre::eyre!(
            "shutdown timed out; some receiver tasks did not stop!"
        ));
    }
    if timeout(Duration::from_secs(SHUTDOWN_TIMEOUT_SECONDS), server_drain)
        .await
        .is_err()
    {
        errors.push(eyre::eyre!(
            "shutdown timed out; some server tasks did not stop!"
        ));
    }
    if timeout(Duration::from_secs(SHUTDOWN_TIMEOUT_SECONDS), task_drain)
        .await
        .is_err()
    {
        errors.push(eyre::eyre!(
            "shutdown timed out; some processing tasks did not stop!"
        ));
    }

    match errors.len() {
        0 => Ok(()),
        1 => Err(errors.remove(0)),
        n => {
            let msgs: Vec<String> = errors.iter().map(|e| format!("{e:#}")).collect();
            Err(eyre::eyre!(
                "{n} errors occured during run/shutdown: {}",
                msgs.join("; ")
            ))
        }
    }
}

// ----- async Task groups -----

fn spawn_rx_tasks(
    udp_addr: SocketAddr,
    receiver_threads: usize,
    state: &AppState,
) -> JoinSet<eyre::Result<()>> {
    // Construct an activity monitor for the network receive tasks
    let monitor = Arc::new(NetworkActivity::new());

    // Spawn multiple listeners
    // let mut rx_tasks: JoinSet<(Result<(), std::io::Error>)> = JoinSet::new();
    let mut rx_tasks = JoinSet::new();

    let receiver_threads = receiver_threads.max(1);
    tracing::info!(recv_threads = ?receiver_threads, "running {receiver_threads} receiver threads");

    for _ in 0..receiver_threads.max(1) {
        rx_tasks.spawn(packet_handler_task(
            udp_addr,
            Arc::clone(&monitor),
            Arc::clone(&state.buffers),
            Arc::clone(&state.metrics),
        ));
    }

    // also spawn the monitor task
    rx_tasks.spawn(packet_monitor_task(
        Arc::clone(&state.buffers),
        Arc::clone(&monitor),
    ));

    rx_tasks
}

fn spawn_proc_tasks(config: Option<AppConfig>, state: &AppState) -> JoinSet<eyre::Result<()>> {
    let mut tasks = JoinSet::new();

    let config: Option<Arc<AppConfig>> = config.map(Arc::new);

    // Start the solar event task
    tasks.spawn(crate::tasks::solar_event_task(
        Arc::clone(&state.metrics),
        config.map(|c| Arc::clone(&c)),
    ));

    // Start the bad input task
    tasks.spawn(crate::tasks::bad_input_task(
        Arc::clone(&state.buffers),
        Arc::clone(&state.computed),
    ));

    tasks
}

async fn spawn_http_tasks(
    http_addr: SocketAddr,
    state: AppState,
) -> eyre::Result<JoinSet<eyre::Result<()>>> {
    let mut tasks = JoinSet::new();

    let http_listener = TcpListener::bind(http_addr)
        .await
        .wrap_err("failed to construct TCP listener")?;

    tasks.spawn(async move {
        axum::serve(http_listener, make_router(state))
            .await
            .map_err(eyre::Report::from)
    });

    tracing::info!(http_addr = ?http_addr, "started HTTP server: {http_addr}");

    Ok(tasks)
}
