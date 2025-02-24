package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"greenlight.alar.net/internal/data"
	"log"
	"net/http"
	"os"
	"time"

	_ "github.com/lib/pq"
)

const DSN = "postgres://greenlight:zxcv@localhost/greenlight?sslmode=disable"
const version = "1.0.0"

type config struct {
	port int
	env  string
	db   struct {
		dsn string
	}
}

type application struct {
	config config
	logger *log.Logger
	models data.Models
}

func main() {
	// instance of the config struct
	var cfg config

	// read the port and env command-line flags into the cfg
	flag.IntVar(&cfg.port, "port", 4000, "API server port")
	flag.StringVar(&cfg.env, "env", "development", "Environment (development|staging|production)")

	flag.StringVar(&cfg.db.dsn, "db-dsn", DSN, "PostgreSQL DSN")

	flag.Parse()

	// logger which writes to the stdout stream,
	// prefixed with the current date and time
	logger := log.New(os.Stdout, "", log.Ldate|log.Ltime)

	db, err := openDB(cfg)
	if err != nil {
		logger.Fatal(err)
	}

	defer db.Close()

	logger.Printf("database connection pool established")

	// instance of application struct
	app := &application{
		config: cfg,
		logger: logger,
		models: data.NewModels(db),
	}

	// mux dispatches requests to the handlers (router)
	// mux := http.NewServeMux()
	// mux.HandleFunc("/v1/healthcheck", app.healthcheckHandler)

	// http server
	srv := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.port),
		Handler:      app.routes(), // mux,
		IdleTimeout:  time.Minute,
		ReadTimeout:  time.Second * 10,
		WriteTimeout: time.Second * 30,
	}

	// log about starting
	logger.Printf("starting %s server on %s", cfg.env, srv.Addr)

	// start the http server
	err = srv.ListenAndServe()
	logger.Fatal(err)
}

func openDB(cfg config) (*sql.DB, error) {
	db, err := sql.Open("postgres", cfg.db.dsn)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = db.PingContext(ctx)
	if err != nil {
		return nil, err
	}

	return db, nil
}
