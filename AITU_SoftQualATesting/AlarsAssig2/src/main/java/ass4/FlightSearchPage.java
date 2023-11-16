package ass4;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class FlightSearchPage {
    WebDriver driver;
    WebDriverWait wait;

    public FlightSearchPage(WebDriver driver, WebDriverWait wait) {
        this.driver = driver;
        this.wait = wait;
    }

    public void enterDeparture(String departure) {
        // Code to enter departure city
        // Find the button using the data-ui-name attribute and click it
        WebElement button_from = driver.findElement(By.cssSelector("button[data-ui-name='input_location_from_segment_0']"));
        button_from.click();
        // Find the input field and enter departure
        WebElement inputField_from = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector("input[data-ui-name='input_text_autocomplete']")));
        inputField_from.sendKeys(departure);
        // Select the first match from the suggestion list
        WebElement firstSuggestion_from = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("ul#flights-searchbox_suggestions > li:first-child")));
        firstSuggestion_from.click();
    }

    public void enterDestination(String destination) {
        // Code to enter destination city
        // Find the button using the data-ui-name attribute and click it
        WebElement button_to = driver.findElement(By.cssSelector("button[data-ui-name='input_location_to_segment_0']"));
        button_to.click();
        // Find the input field and enter destination
        WebElement inputField_to = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector("input[data-ui-name='input_text_autocomplete']")));
        inputField_to.sendKeys(destination);
        // Select the first match from the suggestion list
        WebElement firstSuggestion_to = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("ul#flights-searchbox_suggestions > li:first-child")));
        firstSuggestion_to.click();
    }

    public void selectOneWay() {
        // Click the "One-way" radio button
        WebElement oneWayRadioButton = driver.findElement(By.id("search_type_option_ONEWAY"));
        oneWayRadioButton.click();
    }

    public void submitSearch() {
        // Code to submit the search
        WebElement submitButton = driver.findElement(By.cssSelector("button[data-ui-name='button_search_submit']"));
        submitButton.click();
    }
}
