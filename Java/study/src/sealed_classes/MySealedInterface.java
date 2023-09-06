package sealed_classes;

public sealed interface MySealedInterface permits Alpha, Beta {
    void interface_method();
}
