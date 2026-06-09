//! Integration tests — lifecycle. Split out of `integration.rs`.

mod common;

use common::*;

#[tokio::test]
async fn create_then_get_metadata_and_schema() {
    let app = build_app();

    let body = json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {
                "Item": {
                    "properties": {
                        "name": { "type": "string", "required": true }
                    }
                }
            },
            "edge_types": {}
        }
    });

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);

    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let created: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(created["id"], "g1");
    assert_eq!(created["wire_version"], env!("CARGO_PKG_VERSION"));
    let first_hash = created["content_hash"].as_str().unwrap().to_string();
    assert_eq!(first_hash.len(), 64);
    // POST returns the full schema body (creator wants confirmation of
    // exactly what was stored).
    assert_eq!(created["schema"]["name"], "demo");

    // GET /v1/graphs/{id} is metadata-only — no schema body. Cheap
    // existence check + content-hash drift comparison without the
    // serialized schema on the wire.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let metadata: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(metadata["id"], "g1");
    assert_eq!(metadata["wire_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(metadata["content_hash"].as_str().unwrap(), first_hash);
    assert!(
        metadata.get("schema").is_none(),
        "metadata response must not embed schema; got {metadata}"
    );

    // GET /v1/graphs/{id}/schema returns the full SchemaResponse —
    // the shape generation_plus codegen reads (matches the
    // C-partial `build_schema_contract`).
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let fetched: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(fetched["id"], "g1");
    assert_eq!(fetched["content_hash"].as_str().unwrap(), first_hash);
    assert_eq!(fetched["schema"]["name"], "demo");
    assert_eq!(fetched["schema"]["version"], 1);
}

#[tokio::test]
async fn get_schema_on_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/missing/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn get_unknown_graph_returns_404() {
    let app = build_app();
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/missing")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn malformed_graph_id_rejected_with_400_on_read_and_write_paths() {
    // A malformed id is rejected up front (400), not merely missed
    // (404) — both the read path (`graph_entry`) and the write paths
    // that join the id into a filesystem path on the OnDisk backend
    // (`delete_graph` / `replace_schema`) validate it, matching the
    // create-time contract. `$` is outside the allowed `[A-Za-z0-9_-]`
    // set but routes as a single path segment.
    let app = build_app();
    for method in ["GET", "DELETE"] {
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri("/v1/graphs/bad$id")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::BAD_REQUEST,
            "{method} on a malformed id should be 400"
        );
    }
}

#[tokio::test]
async fn error_responses_use_structured_json_envelope() {
    // Every error — whether it originates in a handler (400) or the
    // registry lookup (404) — carries the same JSON `{ "error": <msg> }`
    // body with an `application/json` content type, so a client parses
    // one shape across all routes.
    let app = build_app();
    for (uri, want_status) in [
        ("/v1/graphs/missing", StatusCode::NOT_FOUND),
        ("/v1/graphs/bad$id", StatusCode::BAD_REQUEST),
    ] {
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), want_status, "for {uri}");
        let content_type = res
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            content_type.starts_with("application/json"),
            "error content-type should be JSON, got {content_type:?} for {uri}"
        );
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).expect("error body must be valid JSON");
        let msg = body["error"].as_str();
        assert!(
            msg.is_some_and(|s| !s.is_empty()),
            "error body must be {{\"error\": <non-empty string>}}, got {body} for {uri}"
        );
    }
}

#[tokio::test]
async fn create_duplicate_graph_returns_409() {
    let app = build_app();
    let body = json!({
        "id": "g1",
        "schema": {
            "name": "demo",
            "version": 1,
            "node_types": {},
            "edge_types": {}
        }
    });
    let make_req = || {
        Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };
    let r1 = app.clone().oneshot(make_req()).await.unwrap();
    assert_eq!(r1.status(), StatusCode::CREATED);
    let r2 = app.clone().oneshot(make_req()).await.unwrap();
    assert_eq!(r2.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn list_graphs_returns_sorted_ids() {
    let app = build_app();
    for id in ["zeta", "alpha", "mike"] {
        let body = json!({
            "id": id,
            "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
        });
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/graphs")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
    }

    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["graphs"], json!(["alpha", "mike", "zeta"]));
}

#[tokio::test]
async fn delete_graph_then_get_returns_404() {
    let app = build_app();
    let create_body = json!({
        "id": "g1",
        "schema": { "name": "demo", "version": 1, "node_types": {}, "edge_types": {} }
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/graphs")
                .header("content-type", "application/json")
                .body(Body::from(create_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NO_CONTENT);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // Second delete is also 404.
    let res = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/graphs/g1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn put_schema_compatible_addition_succeeds_and_hash_changes() {
    let app = build_app_with_item_graph().await;

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let before: Value = serde_json::from_slice(&bytes).unwrap();
    let old_hash = before["content_hash"].as_str().unwrap().to_string();

    // Compatible change: add an optional `nickname` property to Item.
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string", "required": true },
                    "tier": { "type": "string", "default": "standard" },
                    "nickname": { "type": "string" }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item",
                "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let parsed: Value = serde_json::from_str(&body).unwrap();
    let new_hash = parsed["content_hash"].as_str().unwrap();
    assert_ne!(
        new_hash, old_hash,
        "content_hash should change on schema swap"
    );
    // Response embeds the new schema.
    assert!(parsed["schema"]["node_types"]["Item"]["properties"]["nickname"].is_object());

    // Subsequent GET /schema reflects the new shape.
    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let after: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(after["content_hash"].as_str().unwrap(), new_hash);
    assert!(after["schema"]["node_types"]["Item"]["properties"]["nickname"].is_object());
}

#[tokio::test]
async fn put_schema_can_add_new_node_type() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string", "required": true },
                    "tier": { "type": "string", "default": "standard" }
                }
            },
            "Place": {
                "properties": {
                    "name": { "type": "string", "required": true }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item",
                "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
}

#[tokio::test]
async fn put_schema_can_relax_required_to_optional() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string" },
                    "tier": { "type": "string", "default": "standard" }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
}

#[tokio::test]
async fn put_schema_rejects_removed_node_type() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({}),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {}
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("removed node type") && body.contains("Item"),
        "{body}"
    );
}

#[tokio::test]
async fn put_schema_rejects_removed_property() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string", "required": true }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("removed node property") && body.contains("Item.tier"),
        "{body}"
    );
}

#[tokio::test]
async fn put_schema_rejects_changed_property_type() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "int", "required": true },
                    "tier": { "type": "string", "default": "standard" }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("changed node property type"), "{body}");
}

#[tokio::test]
async fn put_schema_rejects_required_without_default_added() {
    let app = build_app_with_item_graph().await;
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "string", "required": true },
                    "tier": { "type": "string", "default": "standard" },
                    "ssn":  { "type": "string", "required": true }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("required without a default") && body.contains("ssn"),
        "{body}"
    );
}

#[tokio::test]
async fn put_schema_lists_all_violations_in_one_response() {
    let app = build_app_with_item_graph().await;
    // Three violations: removed property, changed type on existing,
    // and a required-without-default new property.
    let new = put_schema(
        json!({
            "Item": {
                "properties": {
                    "name": { "type": "int", "required": true },
                    "ssn":  { "type": "string", "required": true }
                }
            }
        }),
        json!({
            "Likes": {
                "from": "Item", "to": "Item",
                "properties": {
                    "weight": { "type": "float" },
                    "source": { "type": "string" }
                }
            }
        }),
    );
    let (status, body) = put_g1_schema(&app, new).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("removed node property"), "{body}");
    assert!(body.contains("changed node property type"), "{body}");
    assert!(body.contains("required without a default"), "{body}");
}

#[tokio::test]
async fn put_schema_on_missing_graph_returns_404() {
    let app = build_app();
    let new = put_schema(json!({}), json!({}));
    let res = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/graphs/nope/schema")
                .header("content-type", "application/json")
                .body(Body::from(new.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn put_schema_failure_leaves_state_intact() {
    // After a rejected PUT, the in-memory schema must be unchanged
    // and existing nodes must still be readable.
    let app = build_app_with_item_graph().await;
    create_item(&app, "n1").await;

    // Submit a breaking PUT.
    let bad = put_schema(json!({}), json!({}));
    let (status, _) = put_g1_schema(&app, bad).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Existing node still readable + the schema still has the
    // original Item type.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/nodes/Item/n1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/graphs/g1/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let schema: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(schema["schema"]["node_types"]["Item"].is_object());
}
