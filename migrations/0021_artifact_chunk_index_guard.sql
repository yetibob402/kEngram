-- no-transaction

-- Session guard for artifact-chunk HNSW/FTS index builds. CREATE INDEX
-- CONCURRENTLY must remain isolated in follow-up files; SQLx applies
-- migrations on one connection, so these GUCs carry to the next builds.

SET lock_timeout = '5s';
SET statement_timeout = '30min';
SET max_parallel_maintenance_workers = 0;
SET maintenance_work_mem = '32MB';
