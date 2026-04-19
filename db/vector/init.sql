CREATE EXTENSION vector;

CREATE TABLE test(id bigserial PRIMARY KEY, embedding vector(3));
INSERT INTO test(embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
