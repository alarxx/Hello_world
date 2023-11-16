package ass5;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class LoggingTask {
    private static final Logger logger = LogManager.getLogger(LoggingTask.class);

    public static void main(String[] args) {
        logger.info("This is an info message");
        logger.error("This is an error message");
        another();
    }

    public static void another(){
        logger.info("This is an info message from another method");
    }
}
