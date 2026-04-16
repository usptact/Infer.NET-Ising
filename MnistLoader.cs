static class MnistLoader
{
    public static (MnistDataset Train, MnistDataset Test) LoadAll(string dataDir) => (
        Load(
            Path.Combine(dataDir, "train-images.idx3-ubyte"),
            Path.Combine(dataDir, "train-labels.idx1-ubyte")),
        Load(
            Path.Combine(dataDir, "t10k-images.idx3-ubyte"),
            Path.Combine(dataDir, "t10k-labels.idx1-ubyte"))
    );

    public static MnistDataset Load(string imagesPath, string labelsPath)
    {
        byte[] labels = ReadLabels(labelsPath);
        byte[][,] images = ReadImages(imagesPath);

        if (labels.Length != images.Length)
            throw new InvalidDataException(
                $"Label count ({labels.Length}) does not match image count ({images.Length}).");

        var samples = new MnistSample[labels.Length];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = new MnistSample(images[i], labels[i]);
        return new MnistDataset(samples);
    }

    public static BinaryMnistSample Binarize(MnistSample sample, byte threshold = 128)
    {
        var pixels = new bool[28, 28];
        for (int row = 0; row < 28; row++)
            for (int col = 0; col < 28; col++)
                pixels[row, col] = sample.Pixels[row, col] >= threshold;
        return new BinaryMnistSample(pixels, sample.Label);
    }

    public static BinaryMnistSample[] Binarize(IReadOnlyList<MnistSample> samples, byte threshold = 128)
    {
        var result = new BinaryMnistSample[samples.Count];
        for (int i = 0; i < samples.Count; i++)
            result[i] = Binarize(samples[i], threshold);
        return result;
    }

    public static void Print(BinaryMnistSample sample)
    {
        Console.WriteLine($"Label: {sample.Label}");
        for (int row = 0; row < 28; row++)
        {
            for (int col = 0; col < 28; col++)
                Console.Write(sample.Pixels[row, col] ? '*' : '.');
            Console.WriteLine();
        }
    }

    // --- private helpers ---

    static byte[] ReadLabels(string path)
    {
        using var reader = new BinaryReader(File.OpenRead(path));
        int magic = ReadInt32BE(reader);
        if (magic != 2049)
            throw new InvalidDataException(
                $"Labels magic number mismatch: expected 2049, got {magic}.");
        int count = ReadInt32BE(reader);
        return reader.ReadBytes(count);
    }

    static byte[][,] ReadImages(string path)
    {
        using var reader = new BinaryReader(File.OpenRead(path));
        int magic = ReadInt32BE(reader);
        if (magic != 2051)
            throw new InvalidDataException(
                $"Images magic number mismatch: expected 2051, got {magic}.");
        int count = ReadInt32BE(reader);
        int rows  = ReadInt32BE(reader);
        int cols  = ReadInt32BE(reader);

        var images = new byte[count][,];
        for (int i = 0; i < count; i++)
        {
            byte[] raw = reader.ReadBytes(rows * cols);
            var img = new byte[rows, cols];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    img[r, c] = raw[r * cols + c];
            images[i] = img;
        }
        return images;
    }

    static int ReadInt32BE(BinaryReader reader)
    {
        byte[] b = reader.ReadBytes(4);
        if (BitConverter.IsLittleEndian) Array.Reverse(b);
        return BitConverter.ToInt32(b, 0);
    }
}

class MnistDataset
{
    readonly Dictionary<byte, List<MnistSample>> _index;

    public IReadOnlyList<MnistSample> All { get; }

    public MnistDataset(MnistSample[] samples)
    {
        All = samples;
        _index = new Dictionary<byte, List<MnistSample>>();
        foreach (var sample in samples)
        {
            if (!_index.TryGetValue(sample.Label, out var list))
                _index[sample.Label] = list = new List<MnistSample>();
            list.Add(sample);
        }
    }

    public IReadOnlyList<MnistSample> GetByLabel(byte label) =>
        _index.TryGetValue(label, out var list) ? list : [];
}

record MnistSample(byte[,] Pixels, byte Label);
record BinaryMnistSample(bool[,] Pixels, byte Label);
