package org.example;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class Third {
    public static void main( String[] args ) {
        WebDriver driver = new ChromeDriver();

        driver.get("https://flight.kz/");
        driver.manage().timeouts().implicitlyWait(Duration.ofMillis(500));

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10)); // 10 seconds timeout for explicit wait

        WebElement from = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector("[name='origin_name']")
        ));
        from.sendKeys("Астана");

        driver.manage().timeouts().implicitlyWait(Duration.ofMillis(500));

        WebElement to = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector("[name='destination_name']")
        ));
        to.sendKeys("Алматы");

        driver.manage().timeouts().implicitlyWait(Duration.ofMillis(2000));

        WebElement findB = driver.findElement(
                By.cssSelector("#flights-form-whitelabel_ru")
        );
        findB.submit();

        WebElement buy = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector(".ticket-action-button-deeplink.ticket-action-button-deeplink--")
        ));
        buy.click();

        WebElement sex = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='Пол']")
        ));
        from.sendKeys("Мужской");

        WebElement fname = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='Фамилия']")
        ));
        fname.sendKeys("Имя");

        WebElement lname = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='Фамилия']")
        ));
        lname.sendKeys("Фамилия");

        WebElement birthd = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='Дата рождения']")
        ));
        birthd.sendKeys("01.01.1980");

        WebElement doctype = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='Тип документа']")
        ));
        doctype.sendKeys("Удостоверение личности");

        WebElement seri = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='Серия и № документа']")
        ));
        seri.sendKeys("11111111");

        WebElement srokDeistv = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='Срок действия']")
        ));
        srokDeistv.sendKeys("07.03.2024");

        WebElement ID = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='ИИН']")
        ));
        ID.sendKeys("010101010101");

        WebElement email = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='ИИН']")
        ));
        email.sendKeys("asdfasdf@gmail.com");

        WebElement number = wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//*[text()='ИИН']")
        ));
        number.sendKeys("87777777777");

        WebElement submitB = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector(".pl-2.pr-2.f-center-center.w-100.font-size-16.size-6.font-600.t-btn-atomic.cursor-pointer.theme-default.t-btn-v2.bg-17.color-1.bs-8.ripple")
        ));
        submitB.submit();
    }
}
