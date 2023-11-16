package org.example;

import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.FluentWait;
import org.openqa.selenium.support.ui.Wait;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;
import java.util.function.Function;

public class Second {
    public static void main( String[] args ) {
        WebDriver driver = new ChromeDriver();

        driver.get("https://shop.kz/");

        /**
         * Implicit Wait.
         * Configures the implicit wait to 500 milliseconds.
         * The driver will wait up to this time when trying to find an element.
         * */
        driver.manage().timeouts().implicitlyWait(Duration.ofMillis(500));

        /**
         * Explicit Wait
         * Initializes an object for explicit wait with a timeout of 10 seconds
         * */
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10)); // 10 seconds timeout for explicit wait


        /**
         * Fluent wait configuration
         * */
        Wait<WebDriver> f_wait = new FluentWait<WebDriver>(driver)
                .withTimeout(Duration.ofSeconds(30))  // Maximum wait time
                .pollingEvery(Duration.ofSeconds(5))  // Polling interval
                .ignoring(NoSuchElementException.class);  // Ignoring NoSuchElementException

        /**
         * Fluent wait example, to wait for the sign-in button to be clickable
         * */
        WebElement signinButton = f_wait.until(new Function<WebDriver, WebElement>() {
            public WebElement apply(WebDriver driver) {
                return driver.findElement(By.cssSelector(".user-auth-enter-button"));
            }
        });
        signinButton.click();


        /**
         * Explicit wait example, waits for the sign-in button to be clickable and then clicks it
         *
        WebElement signinButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector(".user-auth-enter-button")
        ));
        signinButton.click();*/

        WebElement withpassButton = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector(".block-other-login-methods__password-button")
        ));
        withpassButton.click();


        WebElement loginBox = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector("[autocomplete='username']")
        ));
        loginBox.sendKeys("87780010906");

        WebElement passBox = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector("[autocomplete='current-password']")
        ));
        passBox.sendKeys("123456789");

        passBox.sendKeys(Keys.ENTER);

    }
}
