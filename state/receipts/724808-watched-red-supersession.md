# 724808 watched RED r1 §9.3 + r2 §9
ts=2026-08-02T01:21:57Z
DATABASE_URL_host_only=postgres://***@127.0.0.1:55432/kengram
baseline_mig_sha256=bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
baseline_testmod_sha256=0891395d84de14784edf25439abe4c6405b759f5f50d2bfdc57b344ea388d7d0
baseline_wrapper_sha256=922a7ee9f31912063e240d68a47dd7ed84e2923573c28b275c2f60dea6c10bb4
baseline GREEN ok
## R2-M1_zsd01_catch_removed (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: e3dc017eedcf0f69d972544c68113bbdeaf7125d0e02431e8a982ed647cc0f0e
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R2-M1 restored GREEN
## R2-M2_invalid_status_soft (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 0688c444b286077fbf6e56ad31a93f92dff7203085bce1c694164f6541370ce1
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R2-M2 restored GREEN
## R2-M3_grant_execute_ordinary (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R2-M3 restored GREEN
## R2-M4_naive_framing (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 4a7526bcc86166240fe6f6e14e8f883e24ca9494196ff275f4bb3d72ff55aa33
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R2-M4 restored GREEN
## R2-M5_receipt_key_renamed (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 76cfc4994843ded299d16a0d9d04e02da86b3f2696f890bd1c20b29c8f6d6875
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R2-M5 restored GREEN
## R1-M1_status_predicate_deleted (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 47047013cb33bd3ec604ebe266b46416770eedff78acae454656093c5aa6de32
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M1 restored GREEN
## R1-M2_hash_predicate_deleted (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: eb4c3b9c18fff8db7de6de53f676a8e63c9ccbd16fa11ff68700d15acc375dc0
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M2 restored GREEN
## R1-M3_thought_predicate_deleted (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 1e86ed89304d2ec2e5acff797e4f8a7310004a1e3b883cdad43647364e2c53d5
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M3 restored GREEN
## R1-M4_stable_update_or_true (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 508fbb079b213e78e2a1e8ecd47f184fd22867ac3fd667a04b12949f29789b71
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M4 restored GREEN
## R1-M5_retract_removed (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: b780d023ae7bc4e60036c041e4e2ad4cb63c7d8f1ef804688f2d53e0f7913397
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M5 restored GREEN
## R1-M6_replaces_removed (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: dc0c570da9b3b10c42eb108a66ddbd1665d687988280b1a5a5dd794e663826b1
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M6 restored GREEN
R1-M7 alone not RED; amplifying by nulling emb select
## R1-M7_queue_assertion_removed (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 1c9c35ae119a26bff04e0fc0e50d1b1b365bfb10bbc17b36b9eef4eb52901e5d
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M7 restored GREEN
## R1-M8_exact_content_adoption (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 5240c9b1906fdb3ba6ac80b49ab9f92126507363b2fc9fe6ca933a6cc73e9602
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M8 restored GREEN
## R1-M9_session_guard_removed (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 1cb83793c08f26045edfe0ba5df57a2a41cb403ec57665916c1b50d938b78d8d
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M9 restored GREEN
## R1-M10_synthetic_replay (RED)
- exit: see suite.out
- terminal: FAIL source-event-supersession cargo_rc=101
- mig_sha: 149e673e2e636d71e0b25e8c627d107e066140f349479563987e4e8399093987
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M10 restored GREEN
R1-M12 RED evidence: PASS without cargo execution
## R1-M12_early_pass_marker (RED)
- exit: see suite.out
- terminal: PASS source-event-supersession selected=18 executed=18 failed=0 skipped=0
- mig_sha: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
- restored_mig_sha_expected: bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
R1-M12 restored GREEN

ALL WATCHED REDS COMPLETE
final_mig_sha256=bd43fb7e096b746b437ddb262eebb9ec6e8ca4e3eb188662212e1e7b47a4f1fe
final_matches_baseline=yes
