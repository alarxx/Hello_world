package main

import (
	"fmt"
	"net/http"
)

// (req, res) => {...}
func (app *application) healthcheckHandler(w http.ResponseWriter, r *http.Request) {
	js := `{"status": "available", "environment": %q, "version": %q}`
	js = fmt.Sprintf(js, app.config.env, version)

	w.Header().Set("Content-Type", "application/json" /*; charset=utf-8 //not necessary */)

	w.Write([]byte(js))
}
