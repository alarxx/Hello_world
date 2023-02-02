package data

import "time"

type Movie struct {
	ID          int64     `json:"id"` // struct tag
	CreatedDate time.Time `json:"created_date"`
	Title       string    `json:"title"`
	Year        int32     `json:"year"`
	Runtime     int32     `json:"runtime"`
	Genres      []string  `json:"genres"`
	Version     int32     `json:"version"`
}
