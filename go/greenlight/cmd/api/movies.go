package main

import (
	"fmt"
	"greenlight.alar.net/internal/data"
	"net/http"
	"time"
)

func (app *application) createMovieHandler(w http.ResponseWriter, r *http.Request) {
	var input struct {
		Title   string   `json:"title"`
		Year    int32    `json:"year"`
		Runtime int32    `json:"runtime"`
		Genres  []string `json:"genres"`
	}
	/*
		err := json.NewDecoder(r.Body).Decode(&input)
		if err != nil {
			app.errorResponse(w, r, http.StatusBadRequest, err.Error())
			return
		}

		fmt.Fprintf(w, "%+v\n", input)
	*/
	/*
		// less efficient, 80% more memory
		var input struct {
			Foo string `json:"foo"`
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			app.serverErrorResponse(w, r, err)
			return
		}
		err = json.Unmarshal(body, &input)
		if err != nil {
			app.errorResponse(w, r, http.StatusBadRequest, err.Error())
			return
		}
		fmt.Fprintf(w, "%+v\n", input)
	*/
	err := app.readJSON(w, r, &input)
	if err != nil {
		app.badRequestResponse(w, r, err)
		return
	}

	fmt.Fprintf(w, "%+v\n", input)
}

func (app *application) showMovieHandler(w http.ResponseWriter, r *http.Request) {
	id, err := app.readIDParam(r)
	if err != nil {
		//http.NotFound(w, r)
		app.notFoundResponse(w, r)
		return
	}

	movie := envelope{
		"movie": data.Movie{
			ID:        id,
			CreatedAt: time.Now(),
			Title:     "Casablanca",
			Runtime:   102,
			Genres:    []string{"drama", "romance", "war"},
			Version:   1,
		},
	}

	err = app.writeJSON(w, http.StatusOK, movie, nil)
	if err != nil {
		//app.logger.Print(err)
		//http.Error(w, "The server encountered a problem and could not process your request", http.StatusInternalServerError)
		app.serverErrorResponse(w, r, err)
	}

	//fmt.Fprintf(w, "show the details of movie %d\n", id)
}
