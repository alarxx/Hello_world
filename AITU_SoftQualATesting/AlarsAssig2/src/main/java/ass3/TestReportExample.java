package ass3;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import java.time.Duration;

public class TestReportExample {
    String expectedTitleikman ="ikman.lk - Electronics, Cars,        Property and Jobs in Sri Lanka";
    private static WebDriver driver = null;

    @BeforeTest
    public void setup() {
        driver = new ChromeDriver();
    }

    @Test
    public void test() {
        driver = new ChromeDriver();
        driver.get("https://scholar.google.com/");
        driver.manage().timeouts().implicitlyWait(Duration.ofMillis(500));

        WebElement title = driver.findElement(By.cssSelector("#gs_hp_giants"));

        String actualTitle = title.getText();

        Assert.assertEquals(actualTitle, "Стоя на плечах гигантов");
    }

    @AfterTest
    public void tearDownTest() {

        driver.close();
    }
}