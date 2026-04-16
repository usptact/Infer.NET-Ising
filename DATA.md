# DATA.md — MNIST Data Loading Design

## Data Files

The `data/` directory contains two sets of MNIST files. Use the flat files (dot-separated names); the subdirectories are Kaggle packaging artefacts and contain identical content.

| Split | Images file | Labels file |
|-------|-------------|-------------|
| Train (60 000) | `data/train-images.idx3-ubyte` | `data/train-labels.idx1-ubyte` |
| Test  (10 000) | `data/t10k-images.idx3-ubyte`  | `data/t10k-labels.idx1-ubyte`  |

---

## IDX File Format

Both file types use the IDX binary format. All multi-byte integers are **big-endian** (must be byte-reversed on .NET, which is little-endian).

### Labels file (`idx1-ubyte`)

| Offset | Size | Value | Description |
|--------|------|-------|-------------|
| 0 | 4 B | `2049` | Magic number |
| 4 | 4 B | N | Number of items |
| 8 | N × 1 B | 0–9 | Label bytes, one per sample |

### Images file (`idx3-ubyte`)

| Offset | Size | Value | Description |
|--------|------|-------|-------------|
| 0  | 4 B | `2051` | Magic number |
| 4  | 4 B | N | Number of images |
| 8  | 4 B | 28 | Rows |
| 12 | 4 B | 28 | Cols |
| 16 | N × 784 B | 0–255 | Pixel bytes, row-major, one byte per pixel |

---

## C# Design

### Types

```csharp
// Raw grayscale sample, as loaded from disk
record MnistSample(byte[,] Pixels, byte Label);   // Pixels: [row, col], values 0–255

// Binarized sample ready for the Ising model
record BinaryMnistSample(bool[,] Pixels, byte Label); // true = foreground (ink)
```

`byte[,]` (2-D array indexed `[row, col]`) mirrors the Python `img.reshape(28, 28)` layout.

### MnistDataset (class)

Wraps a loaded split and maintains a label index for O(1) lookup by digit.

```csharp
class MnistDataset
{
    // All samples in the split, in file order.
    IReadOnlyList<MnistSample> All { get; }

    // Returns all samples whose Label == label (0–9).
    // Returns an empty list for labels with no samples.
    IReadOnlyList<MnistSample> GetByLabel(byte label);
}
```

The index is a `Dictionary<byte, List<MnistSample>>` built once in the constructor by iterating `All`. No deferred loading or external state.

### MnistLoader (static class)

```csharp
static class MnistLoader
{
    // Load a full split (images + labels from paired files) into a MnistDataset.
    // Throws InvalidDataException on magic-number mismatch.
    static MnistDataset Load(string imagesPath, string labelsPath);

    // Convenience: load both train and test splits from a directory.
    // Expects the four flat files listed above to be present in dataDir.
    static (MnistDataset Train, MnistDataset Test) LoadAll(string dataDir);

    // Binarize a raw sample: pixel >= threshold → true.
    // Recommended threshold: 128.
    static BinaryMnistSample Binarize(MnistSample sample, byte threshold = 128);

    // Binarize a collection of raw samples.
    static BinaryMnistSample[] Binarize(IReadOnlyList<MnistSample> samples, byte threshold = 128);
}
```

### Reading Strategy

Use `BinaryReader` over a `FileStream`. Since IDX is big-endian and .NET is little-endian, read each 4-byte integer with:

```csharp
int ReadBigEndianInt32(BinaryReader r)
{
    byte[] b = r.ReadBytes(4);
    if (BitConverter.IsLittleEndian) Array.Reverse(b);
    return BitConverter.ToInt32(b, 0);
}
```

Read all pixel bytes for a single image in one `ReadBytes(rows * cols)` call, then copy into the `[row, col]` array in a nested loop.

### Binarization

The Ising model requires `bool[,]` inputs. Apply a threshold (default 128) per pixel:

```
pixel >= 128  →  true  (ink / foreground)
pixel <  128  →  false (background)
```

Binarization is a separate step from loading so that callers who need raw grayscale values (e.g. for visualisation or a different model) can obtain them.

### ASCII Visualisation

Add a static helper to `MnistLoader` for printing a binarized image to the console:

```csharp
// Print a 28×28 binary image to stdout. '.' = false (background), '*' = true (ink).
// Label is printed as a header line: "Label: 7"
static void Print(BinaryMnistSample sample);
```

Implementation sketch:

```csharp
static void Print(BinaryMnistSample sample)
{
    Console.WriteLine($"Label: {sample.Label}");
    for (int row = 0; row < 28; row++)
    {
        for (int col = 0; col < 28; col++)
            Console.Write(sample.Pixels[row, col] ? '*' : '.');
        Console.WriteLine();
    }
}
```

Each row is written as a single 28-character string followed by a newline, producing a 29-line output block (1 header + 28 pixel rows).

### Error Handling

Throw `InvalidDataException` (with a message quoting the expected vs actual magic number) if either magic number does not match. No other validation is needed; the files are fixed-size and well-formed.

---

## File Paths at Runtime

The project runs from the repo root, so relative paths work directly:

```csharp
var (train, test) = MnistLoader.LoadAll("data");
```

---

## Usage Example (intended)

```csharp
var (train, test) = MnistLoader.LoadAll("data");

// Query by label
IReadOnlyList<MnistSample> eights = train.GetByLabel(8);
BinaryMnistSample[] binaryEights = MnistLoader.Binarize(eights);

// Pick one image for inference
bool[,] noisyImage = AddNoise(binaryEights[0].Pixels, epsilon: 0.1);
// → feed into Ising model
```
