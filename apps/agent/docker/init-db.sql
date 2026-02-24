-- Creates the agentforge database alongside the default ghostfolio database
-- Runs automatically on first postgres container start

CREATE DATABASE agentforge;

GRANT ALL PRIVILEGES ON DATABASE agentforge TO postgres;
GRANT ALL PRIVILEGES ON DATABASE ghostfolio TO postgres;
