Console.WriteLine("Loading MNIST dataset...");
var (train, _) = MnistLoader.LoadAll("data");
Console.WriteLine($"Loaded {train.All.Count} training samples.");
Console.WriteLine();

PrintFirst3(train, label: 8);
PrintFirst3(train, label: 5);

static void PrintFirst3(MnistDataset dataset, byte label)
{
    var samples = dataset.GetByLabel(label);
    Console.WriteLine($"Found {samples.Count} samples with label {label}. Printing first 3:");
    Console.WriteLine();
    foreach (var sample in samples.Take(3))
    {
        MnistLoader.Print(MnistLoader.Binarize(sample));
        Console.WriteLine();
    }
}
