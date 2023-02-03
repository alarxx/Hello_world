package main

import (
	"fmt"
	"net/http"
)

// Зачем здесь request
func (app *application) logError(r *http.Request, err error) {
	app.logger.Print(err)
}

func (app *application) errorResponse(w http.ResponseWriter, r *http.Request, status int, message any) {
	err := app.writeJSON(w, status, envelope{"error": message}, nil)
	if err != nil {
		app.logError(r, err)
		w.WriteHeader(500)
	}
}

// 500
func (app *application) serverErrorResponse(w http.ResponseWriter, r *http.Request, err error) {
	app.logError(r, err)
	message := "The server encountered a problem and could not process your request"
	app.errorResponse(w, r, http.StatusInternalServerError, message)
}

// 404
func (app *application) notFoundResponse(w http.ResponseWriter, r *http.Request) {
	message := "The request resource could not be found"
	app.errorResponse(w, r, http.StatusNotFound, message)
}

// 405
func (app *application) methodNotAllowedResponse(w http.ResponseWriter, r *http.Request) {
	message := fmt.Sprintf("The %s method in not supported for this resource", r.Method)
	app.errorResponse(w, r, http.StatusMethodNotAllowed, message)
}

func (app *application) badRequestResponse(w http.ResponseWriter, r *http.Request, err error) {
	app.errorResponse(w, r, http.StatusBadRequest, err.Error())
}
