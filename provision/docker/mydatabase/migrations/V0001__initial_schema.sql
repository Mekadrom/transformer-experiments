CREATE SCHEMA myschema;

CREATE TABLE myschema.projects (
    project_key SERIAL,
    name VARCHAR(255) NOT NULL,
    description VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (project_key)
);

CREATE TABLE myschema.datasets (
    dataset_key SERIAL,
    project_key BIGINT NOT NULL,
    name VARCHAR(255) NOT NULL,
    description VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (dataset_key),
    FOREIGN KEY (project_key) REFERENCES myschema.projects(project_key)
);

CREATE TABLE myschema.lexicons (
    lexicon_key SERIAL,
    project_key BIGINT NOT NULL,
    src_language VARCHAR(4) NOT NULL,
    tgt_language VARCHAR(4) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (lexicon_key),
    FOREIGN KEY (project_key) REFERENCES myschema.projects(project_key)
);

CREATE TABLE myschema.hyperparameter_configurations (
    hyperparameters_key SERIAL,
    project_key BIGINT NOT NULL,
    name VARCHAR(255) NOT NULL,
    description VARCHAR(255) NOT NULL,
    data TEXT NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (hyperparameters_key),
    FOREIGN KEY (project_key) REFERENCES myschema.projects(project_key)
);

CREATE TABLE myschema.trained_models (
    model_key SERIAL,
    project_key BIGINT NOT NULL,
    hyperparameters_key BIGINT NOT NULL,
    name VARCHAR(255) NOT NULL,
    description VARCHAR(255) NOT NULL,
    data TEXT NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (model_key),
    FOREIGN KEY (project_key) REFERENCES myschema.projects(project_key),
    FOREIGN KEY (hyperparameters_key) REFERENCES myschema.hyperparameter_configurations(hyperparameters_key)
);

INSERT INTO ops (op) VALUES ('migration V0001__initial_schema.sql');