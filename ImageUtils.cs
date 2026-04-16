/// <summary>
/// Static helpers for binary image manipulation and console rendering.
/// </summary>
public static class ImageUtils
{
    // ── Cropping ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Crop a centred <paramref name="size"/> × <paramref name="size"/> region
    /// from <paramref name="image"/>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="size"/> exceeds either image dimension.
    /// </exception>
    public static bool[,] CropCenter(bool[,] image, int size)
    {
        int h = image.GetLength(0);
        int w = image.GetLength(1);
        if (size > h || size > w)
            throw new ArgumentException(
                $"Crop size {size} exceeds image dimensions {h}×{w}.", nameof(size));

        int rowOffset = (h - size) / 2;
        int colOffset = (w - size) / 2;
        var crop = new bool[size, size];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                crop[i, j] = image[rowOffset + i, colOffset + j];
        return crop;
    }

    // ── Noise ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Add binary symmetric noise by independently flipping each pixel with
    /// probability <paramref name="epsilon"/>.
    /// </summary>
    /// <param name="image">Source image (<c>true</c> = foreground pixel).</param>
    /// <param name="epsilon">Per-pixel flip probability in (0, 0.5).</param>
    /// <param name="rng">
    /// Random number generator to use.  Pass a seeded instance for
    /// reproducibility; defaults to <see cref="Random.Shared"/> when <c>null</c>.
    /// </param>
    /// <returns>A new array with noisy pixels; the source array is unchanged.</returns>
    public static bool[,] AddBinaryNoise(bool[,] image, double epsilon, Random? rng = null)
    {
        rng ??= Random.Shared;
        int h = image.GetLength(0);
        int w = image.GetLength(1);
        var noisy = new bool[h, w];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                noisy[i, j] = rng.NextDouble() < epsilon ? !image[i, j] : image[i, j];
        return noisy;
    }

    // ── Inference utilities ──────────────────────────────────────────────────

    /// <summary>
    /// Threshold a posterior marginal array into a MAP binary image.
    /// Pixels with P(X = true | Y) ≥ <paramref name="threshold"/> map to
    /// <c>true</c>; all others map to <c>false</c>.
    /// </summary>
    /// <param name="posteriors">Posterior marginals from <c>IsingModel.Infer</c>.</param>
    /// <param name="threshold">Decision boundary (default 0.5).</param>
    public static bool[,] Threshold(double[,] posteriors, double threshold = 0.5)
    {
        int h = posteriors.GetLength(0);
        int w = posteriors.GetLength(1);
        var map = new bool[h, w];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                map[i, j] = posteriors[i, j] >= threshold;
        return map;
    }

    /// <summary>
    /// Count pixel-level errors between two same-sized binary images.
    /// </summary>
    public static int CountErrors(bool[,] reference, bool[,] candidate)
    {
        int h = reference.GetLength(0);
        int w = reference.GetLength(1);
        if (candidate.GetLength(0) != h || candidate.GetLength(1) != w)
            throw new ArgumentException("Image dimensions must match.");

        int errors = 0;
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                if (reference[i, j] != candidate[i, j]) errors++;
        return errors;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>
    /// Print min, max, and mean of a posterior array so numerical values are
    /// visible even when the shaded display is hard to read.
    /// </summary>
    public static void PrintStats(double[,] posteriors, string? header = null)
    {
        if (header is not null) Console.WriteLine(header);
        double min = double.MaxValue, max = double.MinValue, sum = 0;
        int count = posteriors.GetLength(0) * posteriors.GetLength(1);
        for (int i = 0; i < posteriors.GetLength(0); i++)
            for (int j = 0; j < posteriors.GetLength(1); j++)
            {
                double v = posteriors[i, j];
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
            }
        Console.WriteLine($"  posterior: min={min:F3}  max={max:F3}  mean={sum / count:F3}");
    }

    // ── Console rendering ─────────────────────────────────────────────────────

    /// <summary>
    /// Render a binary image to stdout.
    /// <c>true</c> pixels are drawn as <c>█</c>; <c>false</c> pixels as a space.
    /// </summary>
    /// <param name="image">Image to render.</param>
    /// <param name="header">Optional header line printed before the image.</param>
    public static void PrintBinary(bool[,] image, string? header = null)
    {
        if (header is not null) Console.WriteLine(header);
        int h = image.GetLength(0);
        int w = image.GetLength(1);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
                Console.Write(image[i, j] ? '█' : ' ');
            Console.WriteLine();
        }
    }

    /// <summary>
    /// Render posterior marginals to stdout using Unicode block-shading characters.
    /// Higher probability → denser block.
    /// <code>
    ///   0.0–0.2 →  (space)
    ///   0.2–0.4 → ░
    ///   0.4–0.6 → ▒
    ///   0.6–0.8 → ▓
    ///   0.8–1.0 → █
    /// </code>
    /// </summary>
    /// <param name="posteriors">Posterior marginals from <c>IsingModel.Infer</c>.</param>
    /// <param name="header">Optional header line printed before the image.</param>
    public static void PrintPosteriors(double[,] posteriors, string? header = null)
    {
        if (header is not null) Console.WriteLine(header);
        int h = posteriors.GetLength(0);
        int w = posteriors.GetLength(1);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
                Console.Write(ToShadeChar(posteriors[i, j]));
            Console.WriteLine();
        }
    }

    private static char ToShadeChar(double p) => p switch
    {
        >= 0.8 => '█',
        >= 0.6 => '▓',
        >= 0.4 => '▒',
        >= 0.2 => '░',
        _      => ' ',
    };
}
