-- To create the database use this commands
CREATE DATABASE DocumentDatabase;
USE DocumentDatabase;

-- To delete the database use this commands
-- DROP TABLE IF EXISTS Documents, Terms, Frequencies;
-- but be careful, this will delete all the data in the database

CREATE TABLE Documents (
    doc_id INT AUTO_INCREMENT PRIMARY KEY,
    doc_url VARCHAR(255),
    doc_title VARCHAR(255),
    doc_text TEXT
);

CREATE TABLE Terms (
    term_id INT AUTO_INCREMENT PRIMARY KEY,
    term_text VARCHAR(255)
);

CREATE TABLE Frequencies (
    doc_id INT,
    term_id INT,
    frequency INT,
    FOREIGN KEY (doc_id) REFERENCES Documents(doc_id),
    FOREIGN KEY (term_id) REFERENCES Terms(term_id),
    PRIMARY KEY (doc_id, term_id)
);


