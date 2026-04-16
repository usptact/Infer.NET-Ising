using System.Diagnostics;

// Ensure the terminal renders Unicode block characters correctly.
Console.OutputEncoding = System.Text.Encoding.UTF8;

// ── Parse arguments ───────────────────────────────────────────────────────────
// Usage: infernet-ising [digit] [index]
//   digit   Digit class to denoise (0–9, default: 8)
//   index   Sample index within that digit class (default: 0)

if (args.Length > 2)
{
    Console.Error.WriteLine("Usage: infernet-ising [digit] [index]");
    Console.Error.WriteLine("  digit   0–9  (default: 8)");
    Console.Error.WriteLine("  index   sample index within that class (default: 0)");
    return 1;
}

byte targetLabel = 8;
if (args.Length >= 1)
{
    if (!byte.TryParse(args[0], out targetLabel) || targetLabel > 9)
    {
        Console.Error.WriteLine($"Error: digit must be an integer 0–9, got '{args[0]}'.");
        return 1;
    }
}

int sampleIndex = 0;
if (args.Length >= 2)
{
    if (!int.TryParse(args[1], out sampleIndex) || sampleIndex < 0)
    {
        Console.Error.WriteLine($"Error: index must be a non-negative integer, got '{args[1]}'.");
        return 1;
    }
}

// ── 1. Load MNIST ─────────────────────────────────────────────────────────────
Console.WriteLine("Loading MNIST dataset...");
var (train, _) = MnistLoader.LoadAll("data");
Console.WriteLine($"Loaded {train.All.Count} training images.");
Console.WriteLine();

// ── 2. Validate digit and index against the loaded dataset ────────────────────
var classImages = train.GetByLabel(targetLabel);
if (classImages.Count == 0)
{
    Console.Error.WriteLine($"Error: no training images found for digit {targetLabel}.");
    return 1;
}
if (sampleIndex >= classImages.Count)
{
    Console.Error.WriteLine(
        $"Error: index {sampleIndex} is out of range — " +
        $"digit {targetLabel} has {classImages.Count} training images (indices 0–{classImages.Count - 1}).");
    return 1;
}

// ── 3. Binarize, crop, add noise ──────────────────────────────────────────────
const double Epsilon  = 0.15;
const double Beta     = 0.5;  // sigmoid(2*0.5)≈0.731 < EP phase-transition threshold ≈0.789
const int    CropSize = 10;

bool[,] full  = MnistLoader.Binarize(classImages[sampleIndex]).Pixels;
bool[,] clean = ImageUtils.CropCenter(full, CropSize);
bool[,] noisy = ImageUtils.AddBinaryNoise(clean, Epsilon, new Random(42));

// ── 4. Display clean and noisy ────────────────────────────────────────────────
Console.WriteLine($"Digit: {targetLabel}  Sample index: {sampleIndex} / {classImages.Count - 1}");
Console.WriteLine();
ImageUtils.PrintBinary(clean, $"=== Clean center crop {CropSize}×{CropSize} ===");
Console.WriteLine();
ImageUtils.PrintBinary(noisy, $"=== Noisy (ε = {Epsilon}) ===");
int noisyErrors = ImageUtils.CountErrors(clean, noisy);
Console.WriteLine($"Pixels corrupted by noise: {noisyErrors} / {CropSize * CropSize}");
Console.WriteLine();

// ── 5. Build model ────────────────────────────────────────────────────────────
Console.WriteLine($"Building Ising model  β = {Beta}, ε = {Epsilon}, grid = {CropSize}×{CropSize}");
var model = new IsingModel(height: CropSize, width: CropSize, beta: Beta, epsilon: Epsilon);

// ── 6. Run inference ──────────────────────────────────────────────────────────
Console.WriteLine("Running EP inference — first call compiles the model...");
var sw = Stopwatch.StartNew();
double[,] posteriors = model.Infer(noisy);
sw.Stop();
Console.WriteLine($"Inference complete in {sw.Elapsed.TotalSeconds:F1} s.");
Console.WriteLine();

// ── 7. Display results ────────────────────────────────────────────────────────
bool[,] denoised = ImageUtils.Threshold(posteriors);
int denoisedErrors = ImageUtils.CountErrors(clean, denoised);

ImageUtils.PrintStats(posteriors, "=== Posterior statistics ===");
Console.WriteLine();
ImageUtils.PrintBinary(denoised, "=== Denoised MAP estimate ===");
Console.WriteLine($"Remaining errors after denoising: {denoisedErrors} / {CropSize * CropSize}");
Console.WriteLine();
ImageUtils.PrintPosteriors(posteriors, "=== Posterior P(X = 1 | Y) ===");

return 0;
