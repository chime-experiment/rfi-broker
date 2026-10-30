//! Router for HTTP endpoints.

use axum::{Router, routing::get};
#[cfg(debug_assertions)]
use {
    axum::middleware::{self, Next},
    axum::routing::post,
};

use super::endpoints;
use crate::state::AppState;

/// Middleware to emit a debug message every time an endpoint is triggered.
///
/// Only available in debug build.
#[cfg(debug_assertions)]
async fn debug_log_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: Next,
) -> axum::response::Response {
    tracing::debug!(
        method = %req.method(),
        uri = %req.uri(),
        "-> request:"
    );

    let response = next.run(req).await;

    tracing::debug!(status = %response.status(), "<- response:");

    response
}

/// Builds a [`Router`] containing all the endpoints we'd like to enable.
///
/// `state` and `metrics` are injected as Axum [`axum::extract::State`]s so handlers
/// can read them.
pub fn make_router(state: AppState) -> Router {
    // router containing all release endpoints
    let router = Router::new()
        .route("/metadata", get(endpoints::metadata))
        .route("/metrics", get(endpoints::metrics))
        .route("/human-metrics", get(endpoints::human_metrics))
        .route(
            "/bad_input_likelihood",
            get(endpoints::get_bad_input_likelihood),
        )
        .route("/", get(endpoints::dump_bad_input_likelihood));

    #[cfg(debug_assertions)]
    let router = router
        .route("/last-frame", get(endpoints::last_frame))
        .route("/write-buffers", post(endpoints::write_buffers));

    let router = router.with_state(state);

    #[cfg(debug_assertions)]
    // enable middleware to log every endpoint request
    let router = router.layer(middleware::from_fn(debug_log_middleware));

    router
}
