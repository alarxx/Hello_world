package data

import "time"

type Movie struct {
	ID          int64
	CreatedDate time.Time
	Title       string
	Year        int32
	Runtime     int32
	Genres      []string
	Version     int32
}
