package ass3;

import org.openqa.selenium.By;
import org.openqa.selenium.NoSuchElementException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.*;

import java.time.Duration;
import java.util.function.Function;

import org.openqa.selenium.interactions.Actions;

public class Main {
    public static void main(String[] args) {
        // Create a new instance of the Chrome driver
        WebDriver driver = new ChromeDriver();

        // Navigate to the web page with the dropdown element
        driver.get("https://the-internet.herokuapp.com/dropdown");

        // Locate the dropdown element
        Select dropdown = new Select(driver.findElement(By.id("dropdown")));

        // Select an option by visible text
        dropdown.selectByVisibleText("Option 1");

//        // Select an option by index
//        dropdown.selectByIndex(2);
//
//        // Select an option by value
//        dropdown.selectByValue("1");

        // Close the browser
    }
}
