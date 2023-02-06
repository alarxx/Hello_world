package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
	"log"
	"net/http"
	"os"
	"time"
)

/* PostgreSQL Database
- - - Basic steps - - -
$ sudo apt install postgresql postgresql-contrib
$ sudo -u postgres psql
(psql) SELECT current_user;
(psql) CREATE DATABASE greenlight;
(psql) \c greenlight
(psql) CREATE ROLE greenlight WITH LOGIN PASSWORD 'password';
(psql) CREATE EXTENSION IF NOT EXISTS citext;
(psql) \q

In order to connect to database we can use DSN or write with flags:
$ psql --host=localhost --dbname=greenlight --username=greenlight

DSN looks like: postgres://greenlight:password@localhost/greenlight

- - - Install pq lib as a PostgreSQL driver - - -
$ go get github.com/lib/pq
To connect to database we have to use DSN

- - - SQL-Migrations - - -
$ curl -L https://github.com/golang-migrate/migrate/releases/download/v4.14.1/migrate.linux-amd64.tar.gz | tar xvz
$ mv migrate.linux-amd64 $GOPATH/bin/migrate
$ migrate -version

$ migrate create -seq -ext=.sql -dir=./migrations migration_name
$ migrate -path=./migrations -database=$DSN up

To check the changes
$ psql $DSN
$ \dt or \d table

migrate methods:
	up (_ or n),
	down(_ or n),
	goto v,
	force v
*/

const version = "1.0.0"

type config struct {
	port int
	env  string
	db   struct {
		dsn          string
		maxOpenConns int
		maxIdleConns int
		maxIdleTime  string
	}
}

type application struct {
	config config
	logger *log.Logger
}

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	var cfg config

	flag.IntVar(&cfg.port, "port", 4000, "API server port")
	flag.StringVar(&cfg.env, "env", "development", "Environment (development | staging | production)")

	flag.StringVar(&cfg.db.dsn, "db-dsn", os.Getenv("GREENLIGHT_DB_DSN"), "PostgreSQL DSN")

	flag.IntVar(&cfg.db.maxOpenConns, "db-max-open-conns", 25, "PostgreSQL max open connections")
	flag.IntVar(&cfg.db.maxIdleConns, "db-max-idle-conns", 25, "PostgreSQL max open connections")
	flag.StringVar(&cfg.db.maxIdleTime, "db-max-idle-time", "15m", "PostgreSQL max open connections")

	flag.Parse()

	logger := log.New(os.Stdout, "", log.Ldate|log.Ltime)

	db, err := openDB(cfg)
	if err != nil {
		logger.Fatal(err)
	}
	defer db.Close()
	logger.Printf("database connection pool established")

	app := &application{
		config: cfg,
		logger: logger,
	}

	srv := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.port),
		Handler:      app.routes(),
		IdleTimeout:  time.Minute,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	logger.Printf("Starting %s server on %s", cfg.env, srv.Addr)
	err = srv.ListenAndServe()
	logger.Fatal(err)
}

func openDB(cfg config) (*sql.DB, error) {
	db, err := sql.Open("postgres", cfg.db.dsn)
	if err != nil {
		return nil, err
	}

	db.SetMaxOpenConns(cfg.db.maxOpenConns)
	db.SetMaxIdleConns(cfg.db.maxIdleConns)

	duration, err := time.ParseDuration(cfg.db.maxIdleTime)
	if err != nil {
		return nil, err
	}

	db.SetConnMaxIdleTime(duration)

	// If the PingContext() call could not complete successfully in 5 seconds, then it will return an error
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	defer cancel()

	err = db.PingContext(ctx)
	if err != nil {
		return nil, err
	}

	return db, nil
}
