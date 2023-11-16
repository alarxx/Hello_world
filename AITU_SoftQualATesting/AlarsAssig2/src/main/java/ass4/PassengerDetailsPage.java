package ass4;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;

public class PassengerDetailsPage {
    WebDriver driver;
    WebDriverWait wait;

    public PassengerDetailsPage(WebDriver driver, WebDriverWait wait) {
        this.driver = driver;
        this.wait = wait;
    }

    public void book(){
        // Next button
        // data-testid="checkout_extras_inner_next"
        WebElement nextButton1 = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector("div[data-testid='checkout_extras_inner_next'] button")));
        nextButton1.click();
    }

    public void enterPassengerDetails() {
        // Code to enter passenger details
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


    }
}
