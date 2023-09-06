package text_blocks;

public class TextBlocks {
    public static void main(String[] args) {
        String text = """
                Hello my name is Name.
                 What is your name?
                """;
        System.out.print(text);

        text = """
                My name is Name too. \
                On one line. \
                """;

        System.out.print(text);
    }
}
