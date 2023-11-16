package ass4;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class Main {
    public static void main(String[] args) throws Exception{
        WebDriver driver = new ChromeDriver();
        driver.get("https://www.booking.com/flights/");
        driver.manage().timeouts().implicitlyWait(Duration.ofMillis(500));

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        FlightSearchPage flightSearchPage = new FlightSearchPage(driver, wait);
        flightSearchPage.enterDeparture("Astana");
        flightSearchPage.enterDestination("Almaty");
        flightSearchPage.selectOneWay();
        flightSearchPage.submitSearch();

        FlightSelectionPage flightSelectionPage = new FlightSelectionPage(driver, wait);
        flightSelectionPage.selectFlight();

        PassengerDetailsPage passengerDetailsPage = new PassengerDetailsPage(driver, wait);
        passengerDetailsPage.enterPassengerDetails();
        passengerDetailsPage.book();

//        driver.quit();
    }
}
