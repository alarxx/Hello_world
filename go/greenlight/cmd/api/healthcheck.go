package main

import (
	"net/http"
)

func (app *application) healthcheckHandler(w http.ResponseWriter, r *http.Request) {
	/*
		fmt.Fprintln(w, "status: available")
		fmt.Fprintf(w, "environment: %s\n", app.config.env)
		fmt.Fprintf(w, "version: %s\n", version)
	*/

	/*
		js := `{
			"status": "available",
			"environment": %q,
			"version": %q
		}`
		js = fmt.Sprintf(js, app.config.env, version)

		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(js))
	*/

	data := envelope{
		"status":      "available",
		"environment": app.config.env,
		"version":     version,
	}

	err := app.writeJSON(w, http.StatusOK, data, nil)

	if err != nil {
		app.serverErrorResponse(w, r, err)
	}
}
