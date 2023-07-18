-- Create new database name my_database

CREATE DATABASE IF NOT EXISTS my_database;

-- Using my_database

USE my_database;

-- Create history data table for

CREATE TABLE
    history_data (
        id INT UNSIGNED NOT NULL AUTO_INCREMENT,
        created_at TIMESTAMP NOT NULL,
        sentences VARCHAR(300) NOT NULL,
        clean_sentences VARCHAR(300) NOT NULL,
        sentiment VARCHAR(8) NOT NULL,
        sentiment_score FLOAT NOT NULL,
        PRIMARY KEY (id)
    );