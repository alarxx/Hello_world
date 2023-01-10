package main
import (
	"fmt"
	"os"
)
func main(){
	/*err := os.Setenv("GREENLIGHT_DB_DSN", "lol");
	if err != nil {
		fmt.Println("Could not set")
	}*/
	dst := os.Getenv("GREENLIGHT_DB_DSN")
	fmt.Printf("GREENLIGHT_DB_DSN: %s\n", dst)
}