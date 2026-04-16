using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

/// <summary>
/// Ising Markov Random Field for binary image denoising via Expectation Propagation.
///
/// <para>The model has two components:</para>
/// <list type="bullet">
///   <item>An Ising prior where every pair of 4-connected neighbouring pixels is
///   encouraged to agree, controlled by coupling strength <c>β</c>.</item>
///   <item>A per-pixel binary symmetric noise channel: each observed pixel was
///   independently flipped from the clean pixel with probability <c>ε</c>.</item>
/// </list>
///
/// <para>The factor graph is compiled on the first <see cref="Infer"/> call.
/// Subsequent calls with different images reuse the compiled graph and only
/// swap the observed values, so they are much faster.</para>
/// </summary>
public sealed class IsingModel
{
    private readonly int _height;
    private readonly int _width;
    private readonly Variable<bool>[,] _x;   // latent clean-image pixels
    private readonly Variable<bool>[,] _y;   // observed noisy-image pixels
    private readonly InferenceEngine _engine;

    /// <summary>Image height this model accepts, in pixels.</summary>
    public int Height => _height;

    /// <summary>Image width this model accepts, in pixels.</summary>
    public int Width => _width;

    /// <summary>
    /// Build the Ising MRF factor graph for images of size
    /// <paramref name="height"/> × <paramref name="width"/>.
    /// </summary>
    /// <param name="height">Number of pixel rows.</param>
    /// <param name="width">Number of pixel columns.</param>
    /// <param name="beta">
    /// Ising coupling strength β &gt; 0.  Typical range: 0.5–2.5.
    /// Higher values produce smoother reconstructions; above ~2.5 EP convergence
    /// may degrade.
    /// </param>
    /// <param name="epsilon">
    /// Per-pixel noise flip probability, 0 &lt; ε &lt; 0.5.
    /// Should match the rate used when corrupting the image.
    /// </param>
    /// <param name="iterations">Number of EP message-passing iterations (default 30).</param>
    public IsingModel(int height, int width, double beta, double epsilon, int iterations = 30)
    {
        if (height <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");
        if (width <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
        if (beta <= 0)
            throw new ArgumentOutOfRangeException(nameof(beta), "β must be positive.");
        if (epsilon <= 0 || epsilon >= 0.5)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "ε must be in the open interval (0, 0.5).");
        if (iterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(iterations), "Iteration count must be positive.");

        _height = height;
        _width  = width;
        _x = new Variable<bool>[height, width];
        _y = new Variable<bool>[height, width];

        BuildModel(beta, epsilon);

        _engine = new InferenceEngine(new ExpectationPropagation())
        {
            NumberOfIterations = iterations,
            ShowProgress       = false,
        };
    }

    /// <summary>
    /// Attach <paramref name="noisyImage"/> as observations and run EP inference.
    /// </summary>
    /// <param name="noisyImage">
    /// Observed noisy image; dimensions must match the model's height × width.
    /// <c>true</c> = foreground (ink), <c>false</c> = background.
    /// </param>
    /// <returns>
    /// Posterior marginals P(X[i,j] = true | Y) as a <c>double[height, width]</c>
    /// array indexed <c>[row, col]</c>.
    /// </returns>
    /// <remarks>
    /// The first call compiles the Infer.NET model to native C# and JIT-compiles
    /// the result, which typically takes 10–30 seconds.  Subsequent calls with
    /// different images run in seconds.
    /// </remarks>
    public double[,] Infer(bool[,] noisyImage)
    {
        if (noisyImage.GetLength(0) != _height || noisyImage.GetLength(1) != _width)
            throw new ArgumentException(
                $"Image size {noisyImage.GetLength(0)}×{noisyImage.GetLength(1)} " +
                $"does not match model dimensions {_height}×{_width}.",
                nameof(noisyImage));

        for (int i = 0; i < _height; i++)
            for (int j = 0; j < _width; j++)
                _y[i, j].ObservedValue = noisyImage[i, j];

        var posteriors = new double[_height, _width];
        for (int i = 0; i < _height; i++)
            for (int j = 0; j < _width; j++)
                posteriors[i, j] = _engine.Infer<Bernoulli>(_x[i, j]).GetProbTrue();

        return posteriors;
    }

    // ── Model construction ───────────────────────────────────────────────────

    private void BuildModel(double beta, double epsilon)
    {
        DeclareLatentVariables();
        AddIsingCoupling(beta);
        AddObservationModel(epsilon);
    }

    /// <summary>
    /// Each latent pixel gets a flat Bernoulli(0.5) marginal prior.
    /// The spatial structure is supplied entirely by the pairwise coupling factors.
    /// </summary>
    private void DeclareLatentVariables()
    {
        for (int i = 0; i < _height; i++)
            for (int j = 0; j < _width; j++)
                _x[i, j] = Variable.Bernoulli(0.5).Named($"x{i}_{j}");
    }

    /// <summary>
    /// Add 4-connected Ising coupling edges.
    ///
    /// <para>The symmetric potential ψ(a, b) = exp(β·(2a−1)(2b−1)) is encoded as
    /// a soft-equality constraint P(a = b) = σ(2β), where σ is the logistic
    /// sigmoid.  This rewards same-valued neighbours and penalises disagreement
    /// equally for both (0,0) and (1,1) pairs.</para>
    /// </summary>
    private void AddIsingCoupling(double beta)
    {
        double pAgree = 1.0 / (1.0 + Math.Exp(-2.0 * beta));  // sigmoid(2β)

        for (int i = 0; i < _height; i++)           // horizontal edges
            for (int j = 0; j < _width - 1; j++)
                ConstrainNeighbours(_x[i, j], _x[i, j + 1], pAgree);

        for (int i = 0; i < _height - 1; i++)       // vertical edges
            for (int j = 0; j < _width; j++)
                ConstrainNeighbours(_x[i, j], _x[i + 1, j], pAgree);
    }

    /// <summary>
    /// Encode the binary symmetric noise model for every pixel.
    ///
    /// <para>
    ///   P(Y[i,j] = 1 | X[i,j] = true)  = 1 − ε  <br/>
    ///   P(Y[i,j] = 1 | X[i,j] = false) = ε
    /// </para>
    ///
    /// <para>Observation variables are declared here but left unobserved until
    /// <see cref="Infer"/> sets their <c>ObservedValue</c>.</para>
    /// </summary>
    private void AddObservationModel(double epsilon)
    {
        for (int i = 0; i < _height; i++)
            for (int j = 0; j < _width; j++)
            {
                _y[i, j] = Variable.New<bool>().Named($"y{i}_{j}");
                using (Variable.If(_x[i, j]))
                    _y[i, j].SetTo(Variable.Bernoulli(1.0 - epsilon));
                using (Variable.IfNot(_x[i, j]))
                    _y[i, j].SetTo(Variable.Bernoulli(epsilon));
            }
    }

    /// <summary>
    /// Add one Ising coupling factor between two neighbouring latent pixels.
    ///
    /// <para>Encoded as a conditional likelihood:
    /// <list type="bullet">
    ///   <item>When a = true:  P(b = true) ∝ pAgree</item>
    ///   <item>When a = false: P(b = true) ∝ 1 − pAgree</item>
    /// </list>
    /// Both branches agree with probability pAgree, matching the symmetric
    /// Ising potential.</para>
    /// </summary>
    private static void ConstrainNeighbours(Variable<bool> a, Variable<bool> b, double pAgree)
    {
        using (Variable.If(a))
            Variable.ConstrainEqualRandom(b, new Bernoulli(pAgree));
        using (Variable.IfNot(a))
            Variable.ConstrainEqualRandom(b, new Bernoulli(1.0 - pAgree));
    }
}
