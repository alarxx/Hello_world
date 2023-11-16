package ass4;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.util.List;

public class FlightSelectionPage {
    WebDriver driver;
    WebDriverWait wait;

    public FlightSelectionPage(WebDriver driver, WebDriverWait wait) {
        this.driver = driver;
        this.wait = wait;
    }

    public void selectFlight() throws Exception {
        // Find all "See flight" buttons
        List<WebElement> seeFlightButtons = wait.until(ExpectedConditions.presenceOfAllElementsLocatedBy(By.cssSelector("button[data-testid='flight_card_bound_select_flight']")));
        // Click the first "See flight" button
        if (!seeFlightButtons.isEmpty()) {
            seeFlightButtons.get(0).click();
        } else {
            throw new Exception("Flights not found!");
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
    }
}
