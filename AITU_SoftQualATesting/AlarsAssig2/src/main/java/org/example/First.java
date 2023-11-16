package org.example;

import org.openqa.selenium.By;
import org.openqa.selenium.Keys;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;

import java.time.Duration;

public class First {
    public static void main( String[] args ) {
        WebDriver driver = new ChromeDriver();
        driver.get("https://scholar.google.com/");
        driver.manage().timeouts().implicitlyWait(Duration.ofMillis(500));

        WebElement title = driver.findElement(By.cssSelector("#gs_hp_giants"));
        System.out.println(title.getText());

        // Using CSS Selector for search box
        WebElement searchBox = driver.findElement(By.cssSelector("#gs_hdr_tsi"));
        searchBox.sendKeys("Selenium WebDriver");
        searchBox.sendKeys(Keys.ENTER);

        // Using XPath for checking if search results are displayed
        WebElement searchResult = driver.findElement(By.xpath("//div[@class='gs_r gs_or gs_scl']"));
        System.out.println(searchResult);
        if (searchResult.isDisplayed()) {
            System.out.println("Search functionality works correctly.");
        } else {
            System.out.println("Search functionality might have an issue.");
        }

    }
}
