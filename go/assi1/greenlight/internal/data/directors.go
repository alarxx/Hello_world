package data

import (
	"database/sql"
	"errors"
	"github.com/lib/pq"
)

type Director struct {
	ID      int64    `json:"id"`
	Name    string   `json:"name"`
	Surname string   `json:"surname"`
	Awards  []string `json:"awards"`
}

type DirectorModel struct {
	DB *sql.DB
}

// method for inserting a new record in the movies table.
func (d DirectorModel) Insert(director *Director) error {
	query := `
		INSERT INTO directors(id, name, surname, awards)
		VALUES ($1, $2, $3, $4)
		RETURNING id`

	return d.DB.QueryRow(query, &director.ID, &director.Name, &director.Surname, pq.Array(&director.Awards)).Scan(&director.ID)
}

// method for fetching a specific record from the movies table.
func (d DirectorModel) Get(id int64) (*Director, error) {
	if id < 1 {
		return nil, ErrRecordNotFound
	}

	query := `
		SELECT *
		FROM directors
		WHERE id = $1`

	var director Director

	err := d.DB.QueryRow(query, id).Scan(
		&director.ID,
		&director.Name,
		&director.Surname,
		pq.Array(&director.Awards),
	)

	if err != nil {
		switch {
		case errors.Is(err, sql.ErrNoRows):
			return nil, ErrRecordNotFound
		default:
			return nil, err
		}
	}

	return &director, nil

}

// method for updating a specific record in the movies table.
func (d DirectorModel) Update(director *Director) error {
	query := `
		UPDATE directors
		SET name = $2, surname = $3, aware = $4
		WHERE id = $5
		RETURNING id`

	args := []interface{}{
		director.ID,
		director.Name,
		director.Surname,
		pq.Array(director.Awards),
	}

	return d.DB.QueryRow(query, args...).Scan(&director.ID)
}

// method for deleting a specific record from the movies table.
func (d DirectorModel) Delete(id int64) error {
	if id < 1 {
		return ErrRecordNotFound
	}
	// Construct the SQL query to delete the record.
	query := `
		DELETE FROM directors
		WHERE id = $1`

	result, err := d.DB.Exec(query, id)
	if err != nil {
		return err
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}

	if rowsAffected == 0 {
		return ErrRecordNotFound
	}

	return nil
}
