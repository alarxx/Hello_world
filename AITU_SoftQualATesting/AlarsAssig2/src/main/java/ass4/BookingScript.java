package ass4;

import org.openqa.selenium.By;
import org.openqa.selenium.NoSuchElementException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.*;

import java.time.Duration;
import java.util.List;

public class BookingScript {
    public static void main( String[] args ) {
        WebDriver driver = new ChromeDriver();

        driver.manage().timeouts().implicitlyWait(Duration.ofMillis(500));

        driver.get("https://www.booking.com/flights/");


        // Find the button using the data-ui-name attribute and click it
        WebElement button_from = driver.findElement(By.cssSelector("button[data-ui-name='input_location_from_segment_0']"));
        button_from.click();

        // Write "astana" in appeared input and click on first match from list
        // Find the input field and enter "Astana"
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        WebElement inputField_from = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector("input[data-ui-name='input_text_autocomplete']")));
        inputField_from.sendKeys("Astana");

        // Select the first match from the suggestion list
        WebElement firstSuggestion_from = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("ul#flights-searchbox_suggestions > li:first-child")));
        firstSuggestion_from.click();

        // Find the button using the data-ui-name attribute and click it
        WebElement button_to = driver.findElement(By.cssSelector("button[data-ui-name='input_location_to_segment_0']"));
        button_to.click();

        // Write "almaty" in appeared input and click on first match from list
        // Find the input field and enter "almaty"
        WebElement inputField_to = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector("input[data-ui-name='input_text_autocomplete']")));
        inputField_to.sendKeys("Almaty");

        // Select the first match from the suggestion list
        WebElement firstSuggestion_to = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("ul#flights-searchbox_suggestions > li:first-child")));
        firstSuggestion_to.click();

        // Click the "One-way" radio button
        WebElement oneWayRadioButton = driver.findElement(By.id("search_type_option_ONEWAY"));
        oneWayRadioButton.click();

        // Click the submit button
        WebElement submitButton = driver.findElement(By.cssSelector("button[data-ui-name='button_search_submit']"));
        submitButton.click();

        /*Wait<WebDriver> wait = new FluentWait<WebDriver>(driver)
                .withTimeout(Duration.ofSeconds(30))
                .pollingEvery(Duration.ofSeconds(5))
                .ignoring(NoSuchElementException.class);*/

        // Loading and move to the next page with list of flights

        // Find all "See flight" buttons
        List<WebElement> seeFlightButtons = wait.until(ExpectedConditions.presenceOfAllElementsLocatedBy(By.cssSelector("button[data-testid='flight_card_bound_select_flight']")));

        // Click the first "See flight" button
        if (!seeFlightButtons.isEmpty()) {
            seeFlightButtons.get(0).click();
        }

        // Modal window
        // Click the "Select" button within the modal
        // waits until the "Select" button is clickable, and then it finds the button using its data-testid attribute within a div and targeting the button element inside it.
        WebElement selectButton = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("div[data-testid='flight_details_inner_modal_select_button'] button")));
        selectButton.click();

        // Next page "Choose your flexibility"
        // Choosing standard ticket just by clicking next
        WebElement nextButton = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("div[data-testid='checkout_ticket_type_inner_next'] button")));
        nextButton.click();

        // Next page "Who's flying?" form
        WebElement input_booker_email = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector("input[name='booker.email']")));
        input_booker_email.sendKeys("testman@gmail.com");

        WebElement input_phone_number = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector("input[name='number']")));
        input_phone_number.sendKeys("7007000770");

        WebElement input_firstName = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector("input[name='passengers.0.firstName']")));
        input_firstName.sendKeys("Name");

        WebElement input_lastname = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector("input[name='passengers.0.lastName']")));
        input_lastname.sendKeys("Lastname");

        // Find the dropdown using its attributes and select the "Male" option
        WebElement genderDropdown = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("select[name='passengers.0.gender']")));
        Select selectGender = new Select(genderDropdown);
        selectGender.selectByValue("male");

        // Find the dropdown using its attributes and select the "Male" option
        WebElement birthDateDropdown = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("select[name='passengers.0.birthDate']")));
        Select selectBirthDate = new Select(birthDateDropdown);
        // electBirthDate.selectByIndex(1); // we can also select by index
        selectBirthDate.selectByValue("01"); // January

        // DD
        WebElement birthDateDD = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("input[data-testid='traveller_data_field_passenger_0_dd']")));
        birthDateDD.sendKeys("1");
        // YY
        WebElement birthDateYY = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("input[data-testid='traveller_data_field_passenger_0_yyyy']")));
        birthDateYY.sendKeys("1991");

        // Inputting values into the expiration date fields
        WebElement expiry_month_input = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("#passengers\\.0\\.expiryDate input[name='month']")));
        WebElement expiry_day_input = driver.findElement(By.cssSelector("#passengers\\.0\\.expiryDate input[name='day']"));
        WebElement expiry_year_input = driver.findElement(By.cssSelector("#passengers\\.0\\.expiryDate input[name='year']"));

        expiry_month_input.sendKeys("05");  // May
        expiry_day_input.sendKeys("15");    // 15th day
        expiry_year_input.sendKeys("2025"); // Year 2025

        // name="passengers.0.passportNumber"
        WebElement passport_number_input = wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector("input[name='passengers.0.passportNumber']")));
        passport_number_input.sendKeys("123123789789"); // 9 digits number

        // name="passengers.0.nationality"
        WebElement nationalityDropdown = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("select[name='passengers.0.nationality']")));
        Select selectNationality = new Select(nationalityDropdown);
        selectNationality.selectByValue("KZ");

        // Next button
        // data-testid="checkout_extras_inner_next"
        WebElement nextButton1 = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("div[data-testid='checkout_extras_inner_next'] button")));
        nextButton1.click();
    }
}
