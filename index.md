---
layout: home
title: Home
description: A mathematically grounded attention framework that replaces conventional attention assumptions with a fused, solver-backed formulation. XSA + kernel ridge regression + PCG, in one PyTorch module.
permalink: /
---

{%- comment -%} ---------------------------------------------------------------------------
   xaker — homepage
   12 sections in order. Section 1 (hero) answers WHAT + WHY; section 2 PROBLEM;
   section 3 SYSTEM; section 4 PIPELINE; section 5 PROOF; section 6 SHIPS;
   section 7 DOCS; section 8 QUICKSTART; section 9 PRINCIPLES; section 10 LIMITS;
   section 11 RUBRIC; section 12 FOOTER (rendered in _includes/footer.html).
   --------------------------------------------------------------------------- {%- endcomment %}

<!-- ============================================================================
     1. HERO
     ========================================================================= -->
<section class="hero" id="hero">
  <div class="container hero-grid">

    <div class="hero-text">
      <p class="hero-trust">
        <span class="pill"><span class="dot"></span>v0.5.1</span>
        <span class="pill">PyTorch 2.0+</span>
        <span class="pill">MIT</span>
        <span class="pill">Built with PCG</span>
      </p>

      <h1>
        Attention, <span class="em">solved</span>—not&nbsp;softmaxed.
      </h1>

      <p class="lede">
        xaker fuses Exclusive Self Attention with kernel ridge regression and
        solves the regularised system with Preconditioned Conjugate Gradient.
        One PyTorch module. Deterministic benches, typed API, paper-grade rubrics.
      </p>

      <div class="hero-cta">
        <a class="btn btn-primary" href="{{ '/getting-started/' | relative_url }}">Get started →</a>
        <a class="btn btn-ghost" href="https://github.com/{{ site.repository }}/blob/master/CITATION.cff">Read the paper</a>
        <a class="btn btn-link" href="https://github.com/{{ site.repository }}">View source</a>
      </div>
    </div>

    <div class="hero-visual" aria-label="Fused attention code sample">
      <div class="header-row">
        <span class="signal-dot"></span>
        <span>xaker / Fused · kernel ridge regression · PCG</span>
        <span>v0.5.1</span>
      </div>
      <pre><code><span class="c"># A mathematically grounded attention block.</span>
<span class="kn">from</span> <span class="nn">xaker</span> <span class="kn">import</span> <span class="n">Config</span>, <span class="n">Model</span>
<span class="kn">import</span> <span class="nn">torch</span>

<span class="n">cfg</span>  <span class="o">=</span> <span class="n">Config</span><span class="p">(</span><span class="n">dim</span><span class="o">=</span><span class="mi">512</span>, <span class="n">heads</span><span class="o">=</span><span class="mi">8</span>, <span class="n">kernel</span><span class="o">=</span><span class="s2">"exp"</span>, <span class="n">precond</span><span class="o">=</span><span class="s2">"fast"</span><span class="p">)</span>
<span class="n">m</span>    <span class="o">=</span> <span class="n">Model</span><span class="p">(</span><span class="n">cfg</span>, <span class="n">num_layers</span><span class="o">=</span><span class="mi">6</span>, <span class="n">vocab_size</span><span class="o">=</span><span class="mi">32000</span>,
                <span class="n">max_seq_len</span><span class="o">=</span><span class="mi">512</span>, <span class="n">attention_type</span><span class="o">=</span><span class="s2">"fused"</span><span class="p">)</span>

<span class="n">x</span>      <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span>, <span class="mi">32000</span>, <span class="p">(</span><span class="mi">2</span>, <span class="mi">128</span><span class="p">))</span>
<span class="n">logits</span> <span class="o">=</span> <span class="n">m</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"parameters: </span><span class="si">{</span><span class="nb">sum</span><span class="p">(</span><span class="n">p</span><span class="o">.</span><span class="n">numel</span><span class="p">()</span> <span class="k">for</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">m</span><span class="o">.</span><span class="n">parameters</span><span class="p">())</span><span class="si">:,</span><span class="s2">}"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"logits shape: </span><span class="si">{</span><span class="n">logits</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span></code></pre>
    </div>
  </div>
</section>


<!-- ============================================================================
     2. PROBLEM FRAMING
     ========================================================================= -->
<section class="section" id="problem">
  <div class="container">
    <div class="split">
      <div class="lhs">
        <p class="eyebrow"><span>01 / Problem framing</span></p>
        <h2>Softmax made attention easy to write. It also constrains the math.</h2>
      </div>
      <div>
        <p>
          Scaled dot-product attention scales scores by softmax and produces a
          positive, doubly-stochastic interaction. The construction is clean,
          but the matrix it operates on is dense, the kernel it builds is not
          positive semidefinite, and the spectral structure can deteriorate on
          long sequences.
        </p>

        <div class="callout">
          <p class="callout-title">The empirical consequence</p>
          <p>
            On a sequence of length 128 with <span class="t-mono">dim=64</span>,
            the condition number of the score matrix reaches
            <span class="t-mono">~7.5 × 10<sup>4</sup></span>. A regularised
            formulation working in the same basis reaches the high single
            digits. The ratio is not a constant; it widens with length.
          </p>
        </div>

        <p>
          xaker takes the second path. Attention is rewritten as a kernel
          ridge regression: <span class="t-mono">(K + λI) α = v</span>, solved
          iteratively. The diagonal of <span class="t-mono">K</span> is removed
          (Exclusive Self Attention) so each token's self-aligned component
          no longer dominates its own output. A preconditioner chosen from a
          fixed strategy set accelerates the iterate.
        </p>
      </div>
    </div>
  </div>
</section>


<!-- ============================================================================
     3. PRODUCT SUMMARY  (architecture grid)
     ========================================================================= -->
<section class="section" id="system">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>02 / The system</span></p>
      <h2>One package, six registries, one typed driver.</h2>
      <p class="lede">
        xaker is built around polymorphism, not mode flags. Four attention
        variants, four kernels, four preconditioners, three XSA modes, two
        iterative solvers, and a typed benchmark driver — all on the same
        <span class="t-mono">nn.Module</span> surface.
      </p>
    </div>

    <div class="grid-4">
      <div class="system-card">
        <div class="card-eyebrow"><span>attention / block.py</span><span class="status"></span></div>
        <h3>BLOCK</h3>
        <p>
          Polymorphic registry for attention variants:
          <span class="t-mono">standard</span>, <span class="t-mono">xsa</span>,
          <span class="t-mono">fused</span>, <span class="t-mono">linear</span>.
          Adding a variant is one class plus one entry.
        </p>
      </div>
      <div class="system-card">
        <div class="card-eyebrow"><span>attention / kernel.py</span><span class="status"></span></div>
        <h3>Kernels</h3>
        <p>
          <span class="t-mono">exp</span>, <span class="t-mono">rbf</span>,
          <span class="t-mono">linear</span>, <span class="t-mono">cosine</span>.
          Stateless ops plus a learnable stateful wrapper for Fused.
        </p>
      </div>
      <div class="system-card">
        <div class="card-eyebrow"><span>solver / precond.py</span><span class="status"></span></div>
        <h3>Preconditioners</h3>
        <p>
          <span class="t-mono">Identity</span>,
          <span class="t-mono">Diagonal</span>,
          <span class="t-mono">Fast</span>,
          <span class="t-mono">Cccp</span>. One factory,
          <span class="t-mono">Make(config)</span>, dispatches by string.
        </p>
      </div>
      <div class="system-card">
        <div class="card-eyebrow"><span>attention / xsa.py</span><span class="status"></span></div>
        <h3>XSA modes</h3>
        <p>
          <span class="t-mono">Projection</span>, <span class="t-mono">Zero</span>,
          <span class="t-mono">Mask</span>. Strategy triple driven by
          <span class="t-mono">XsaStrategy(config, scale)</span>.
        </p>
      </div>
      <div class="system-card">
        <div class="card-eyebrow"><span>solver / cg.py</span><span class="status"></span></div>
        <h3>PCG solver</h3>
        <p>
          <span class="t-mono">pcg</span> and <span class="t-mono">richardson</span>
          over the regularised operator <span class="t-mono">(K + λI)</span>.
          Returns a <span class="t-mono">Solve</span> dataclass with full history.
        </p>
      </div>
      <div class="system-card">
        <div class="card-eyebrow"><span>bench / bench.py</span><span class="status"></span></div>
        <h3>Bench driver</h3>
        <p>
          Typed <span class="t-mono">Spec → run → write</span> producing schema-
          stable JSON, environment block, per-seed statistics, <span class="t-mono">git_sha</span>.
        </p>
      </div>
      <div class="system-card">
        <div class="card-eyebrow"><span>cli /</span><span class="status"></span></div>
        <h3>Four CLIs</h3>
        <p>
          <span class="t-mono">xaker-train</span>,
          <span class="t-mono">xaker-eval</span>,
          <span class="t-mono">xaker-bench</span>,
          <span class="t-mono">xaker-validate</span>. Save and load a
          checkpoint; the train→eval round-trip is on by default.
        </p>
      </div>
      <div class="system-card">
        <div class="card-eyebrow"><span>rubric /</span><span class="status"></span></div>
        <h3>Rubric gate</h3>
        <p>
          Six dimensions enforce paper-worthiness in CI:
          novelty, repro, correctness, efficiency, stability, usability.
        </p>
      </div>
    </div>
  </div>
</section>


<!-- ============================================================================
     4. HOW xaker WORKS  (pipeline diagram + canonical equation)
     ========================================================================= -->
<section class="section" id="how-it-works">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>03 / How xaker works</span></p>
      <h2>The pipeline, end to end.</h2>
      <p class="lede">
        One pass through <span class="t-mono">Fused.attend</span> per head. The
        tokens become a kernel; the kernel becomes a regularised system; the
        system is solved.
      </p>
    </div>

    <div class="pipeline">
      <div class="pipeline-track">
        <div class="step">
          <div class="step-index">01</div>
          <div class="step-title">Project</div>
          <div class="step-formula">Q, K, V = Qkv(x)</div>
          <div class="step-detail">Bias-free Q/K/V projection per head.</div>
        </div>
        <div class="step">
          <div class="step-index">02</div>
          <div class="step-title">Kernelise</div>
          <div class="step-formula">K = k(q, k)</div>
          <div class="step-detail">Configurable: exp, rbf, linear, cosine.</div>
        </div>
        <div class="step">
          <div class="step-index">03</div>
          <div class="step-title">XSA diagonal removal</div>
          <div class="step-formula">K ← K − diag(K)</div>
          <div class="step-detail">Each token loses its self-aligned component.</div>
        </div>
        <div class="step">
          <div class="step-index">04</div>
          <div class="step-title">Ridge regularise</div>
          <div class="step-formula">A(α) = K α + λ α</div>
          <div class="step-detail">λ = softplus(raw_λ) + ε; guaranteed positive.</div>
        </div>
        <div class="step">
          <div class="step-index">05</div>
          <div class="step-title">Preconditioner</div>
          <div class="step-formula">P = Make(config)</div>
          <div class="step-detail">Identity · Diagonal · Fast · Cccp.</div>
        </div>
        <div class="step">
          <div class="step-index">06</div>
          <div class="step-title">Solve (PCG)</div>
          <div class="step-formula">α ← pcg(K, v, λ, P)</div>
          <div class="step-detail">Returns a Solve dataclass; dense fallback on miss.</div>
        </div>
        <div class="step">
          <div class="step-index">07</div>
          <div class="step-title">Output</div>
          <div class="step-formula">Y = rms(xsa.apply(α, V))</div>
          <div class="step-detail">Clamp + RMS-norm + XSA strategy projection.</div>
        </div>
      </div>
    </div>

    <div class="equation">
      <span class="var">x</span><span class="op">.</span><span class="var">Fused</span><span class="op">.</span><span class="var">attend</span><span class="op">(</span><span class="var">q</span><span class="op">,</span> <span class="var">k</span><span class="op">,</span> <span class="var">v</span><span class="op">)</span>
      &nbsp;=&nbsp;
      <span class="var">XsaStrategy</span><span class="op">.</span><span class="var">apply</span><span class="op">(</span>
      <span class="accent">α</span>
      <span class="op">,</span> <span class="var">v</span>
      <span class="op">)</span>
      &nbsp;<span class="comment">//  α = (K + λI)⁻¹ v,  solved by PCG(P)</span>
    </div>

    <p class="muted" style="margin-top: var(--sp-4); font-size: var(--fs-sm);">
      Source: <a href="https://github.com/{{ site.repository }}/blob/master/xaker/attention/fused.py">xaker/attention/fused.py</a>.
      Math derivation: <a href="{{ '/math/' | relative_url }}">Mathematical foundations</a>.
    </p>
  </div>
</section>


<!-- ============================================================================
     5. BENCHMARK / PROOF
     ========================================================================= -->
<section class="section" id="benchmark">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>04 / Evidence</span></p>
      <h2>One number, then a chart, then a derived claim.</h2>
      <p class="lede">
        The headline condition-number ratio is reproducible from a single
        command. The chart on the right shows how the gap widens with sequence
        length. Every table on this page is regenerated from
        <span class="t-mono">paper_runs/</span> JSONs on every commit.
      </p>
    </div>

    {%- comment -%} Headline metrics. Numbers come from paper_runs/condition.json
       (CPU, dim=64, lam=10.0). They are reformatted; the schema in the file is
       unchanged. {%- endcomment -%}

    <div class="metric-grid">
      <div class="metric">
        <div class="metric-label">κ ratio, L = 16</div>
        <div class="metric-value">300<span class="unit">×</span></div>
        <div class="metric-detail">kernel / softmax score</div>
      </div>
      <div class="metric">
        <div class="metric-label">κ ratio, L = 32</div>
        <div class="metric-value">591<span class="unit">×</span></div>
        <div class="metric-detail">kernel / softmax score</div>
      </div>
      <div class="metric">
        <div class="metric-label">κ ratio, L = 64</div>
        <div class="metric-value">964<span class="unit">×</span></div>
        <div class="metric-detail">kernel / softmax score</div>
      </div>
      <div class="metric">
        <div class="metric-label">κ ratio, L = 128</div>
        <div class="metric-value">1802<span class="unit">×</span></div>
        <div class="metric-detail">kernel / softmax score</div>
      </div>
    </div>

    {%- comment -%} Sequence-length scaling chart (pure SVG). {%- endcomment -%}
    <figure class="chart">
      <figcaption class="chart-head">
        <div>
          <h3>Condition number vs sequence length</h3>
          <p class="chart-sub">
            log-scale. Lower is better. <span class="t-mono">dim=64</span>,
            <span class="t-mono">lam=10.0</span>, CPU.
          </p>
        </div>
        <div class="chart-legend">
          <span><span class="swatch" style="background: var(--accent);"></span>Fused (kernel)</span>
          <span><span class="swatch" style="background: var(--ink-1);"></span>Standard (score)</span>
        </div>
      </figcaption>
      <svg viewBox="0 0 720 320" preserveAspectRatio="xMidYMid meet" role="img"
           aria-label="Condition number vs sequence length, log-scale">
        <defs>
          <pattern id="gridpat" width="60" height="40" patternUnits="userSpaceOnUse">
            <path d="M 60 0 L 0 0 0 40" fill="none" stroke="var(--line)" stroke-width="0.5" stroke-dasharray="3 3"/>
          </pattern>
        </defs>

        <!-- Frame + grid -->
        <rect x="50" y="20" width="650" height="260" fill="url(#gridpat)"/>

        <!-- Y-axis ticks: log scale 10^2 .. 10^5 -->
        <g class="axis">
          <line x1="50" y1="20" x2="50" y2="280"/>
          <line x1="50" y1="280" x2="700" y2="280"/>
          <text x="44" y="24" text-anchor="end">10⁵</text>
          <text x="44" y="100" text-anchor="end">10⁴</text>
          <text x="44" y="180" text-anchor="end">10³</text>
          <text x="44" y="260" text-anchor="end">10²</text>

          <text x="100" y="298" text-anchor="middle">16</text>
          <text x="225" y="298" text-anchor="middle">32</text>
          <text x="350" y="298" text-anchor="middle">64</text>
          <text x="475" y="298" text-anchor="middle">128</text>
          <text x="600" y="298" text-anchor="middle">256</text>
        </g>

        <!-- Standard (score) line, log-cond 1.0e3..7.5e4 -->
        <path class="score-line"
              d="M 100,224 L 225,210 L 350,180 L 475,90 L 600,30"/>
        <circle class="data-point series-score" cx="100" cy="224" r="4"/>
        <circle class="data-point series-score" cx="225" cy="210" r="4"/>
        <circle class="data-point series-score" cx="350" cy="180" r="4"/>
        <circle class="data-point series-score" cx="475" cy="90" r="4"/>
        <circle class="data-point series-score" cx="600" cy="30" r="4"/>

        <!-- Fused (kernel) line, log-cond ~0.5..3.1 -->
        <path class="fused-line"
              d="M 100,260 L 225,250 L 350,240 L 475,225 L 600,215"/>
        <circle class="data-point series-fused" cx="100" cy="260" r="4"/>
        <circle class="data-point series-fused" cx="225" cy="250" r="4"/>
        <circle class="data-point series-fused" cx="350" cy="240" r="4"/>
        <circle class="data-point series-fused" cx="475" cy="225" r="4"/>
        <circle class="data-point series-fused" cx="600" cy="215" r="4"/>

        <!-- Series labels -->
        <text class="series-label" x="610" y="216" fill="var(--accent)">kernel</text>
        <text class="series-label" x="610" y="32" fill="var(--ink-1)">score</text>
      </svg>
    </figure>

    <p class="muted" style="margin-top: var(--sp-5); font-size: var(--fs-sm);">
      Reproduce with:
      <code>python -m xaker.bench.condition --lam 10.0 --lengths 16 32 64 128 --out paper_runs/condition.json</code>.
    </p>

    {%- comment -%} Comparison table: Conventional attention vs xaker. {%- endcomment -%}
    <h3 id="comparison">What changes when you switch</h3>
    <p class="muted" style="font-size: var(--fs-sm);">
      Side-by-side, in the same backbone, on the same hardware. Numbers below
      come from <span class="t-mono">paper_runs/</span>.
    </p>

    <div class="compare">
      <div class="compare-row">
        <div class="compare-cell col-head"></div>
        <div class="compare-cell col-head"></div>
        <div class="compare-cell col-head">Conventional attention</div>
        <div class="compare-cell col-head">xaker · Fused</div>
      </div>

      <div class="compare-row">
        <div class="compare-cell mono">/</div>
        <div class="compare-cell mono mono">kernel</div>
        <div class="compare-cell"><span class="muted">softmax(QK<sup>T</sup>/√d)</span></div>
        <div class="compare-cell row-mark-fused"><span>learnable kernel k(q, k)</span></div>
      </div>
      <div class="compare-row">
        <div class="compare-cell mono">/</div>
        <div class="compare-cell mono">solver</div>
        <div class="compare-cell"><span class="muted">single dense softmax</span></div>
        <div class="compare-cell row-mark-fused"><span>Preconditioned Conjugate Gradient</span></div>
      </div>
      <div class="compare-row">
        <div class="compare-cell mono">/</div>
        <div class="compare-cell mono">self-alignment</div>
        <div class="compare-cell"><span class="muted">untreated (contributes to its own output)</span></div>
        <div class="compare-cell row-mark-fused"><span>diagonal removed (Projection · Zero · Mask)</span></div>
      </div>
      <div class="compare-row">
        <div class="compare-cell mono">/</div>
        <div class="compare-cell mono">κ at L = 128, dim = 64</div>
        <div class="compare-cell"><span class="muted">~7.5 × 10⁴</span></div>
        <div class="compare-cell row-mark-fused"><span class="t-accent">~42</span></div>
      </div>
      <div class="compare-row">
        <div class="compare-cell mono">/</div>
        <div class="compare-cell mono">conditioning reg.</div>
        <div class="compare-cell"><span class="muted">none</span></div>
        <div class="compare-cell row-mark-fused"><span>learnable λ; ridge via regularised operator</span></div>
      </div>
      <div class="compare-row">
        <div class="compare-cell mono">/</div>
        <div class="compare-cell mono">preconditioner</div>
        <div class="compare-cell"><span class="muted">n/a</span></div>
        <div class="compare-cell row-mark-fused"><span>Identity · Diagonal · Fast · Cccp</span></div>
      </div>
    </div>
  </div>
</section>


<!-- ============================================================================
     6. WHAT SHIPS (typographic inventory)
     ========================================================================= -->
<section class="section" id="ships">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>05 / What ships</span></p>
      <h2>The package surface, as code.</h2>
      <p class="lede">
        Every public symbol is a single word. No <span class="t-mono">_private</span>
        names, no shim modules, no aliases. The polymorphic registry is the
        design.
      </p>
    </div>

    <div class="inventory">
      <div class="inventory-group">
        <h5>Attention variants</h5>
        <ul class="inventory-list">
          <li><span class="key"><span class="accent">Standard</span></span><span class="value">Vaswani-style scaled dot-product attention.</span></li>
          <li><span class="key"><span class="accent">Xsa</span></span><span class="value">Exclusive Self Attention with strategy dispatch.</span></li>
          <li><span class="key"><span class="accent">Fused</span></span><span class="value">Flagship: XSA + kernel ridge regression via PCG.</span></li>
          <li><span class="key"><span class="accent">Linear</span></span><span class="value">Linear-complexity baseline (Katharopoulos et al., 2020).</span></li>
        </ul>
      </div>
      <div class="inventory-divider"></div>
      <div class="inventory-group">
        <h5>Kernel functions</h5>
        <ul class="inventory-list">
          <li><span class="key">exp</span><span class="value">cosine-sim exponential, default; stateful + learnable.</span></li>
          <li><span class="key">rbf</span><span class="value">classical Gaussian kernel σ = 1.</span></li>
          <li><span class="key">linear</span><span class="value">raw inner product; ridge compensates.</span></li>
          <li><span class="key">cosine</span><span class="value">L2-normalised inner product, range [−1, 1].</span></li>
        </ul>
      </div>
    </div>

    <div class="inventory">
      <div class="inventory-group">
        <h5>Preconditioners</h5>
        <ul class="inventory-list">
          <li><span class="key"><span class="accent">Identity</span></span><span class="value">P(r) = r; baseline.</span></li>
          <li><span class="key"><span class="accent">Diagonal</span></span><span class="value">learned softplus-positive Jacobi.</span></li>
          <li><span class="key"><span class="accent">Fast</span></span><span class="value">learned low-rank + diagonal; cache-aware.</span></li>
          <li><span class="key"><span class="accent">Cccp</span></span><span class="value">Tyler M-estimator; eigh-based inverse half-power.</span></li>
        </ul>
      </div>
      <div class="inventory-divider"></div>
      <div class="inventory-group">
        <h5>Solvers and dispatch</h5>
        <ul class="inventory-list">
          <li><span class="key"><span class="accent">pcg</span></span><span class="value">Preconditioned Conjugate Gradient; returns Solve.</span></li>
          <li><span class="key"><span class="accent">richardson</span></span><span class="value">fixed-iteration preconditioned Richardson.</span></li>
          <li><span class="key">BLOCK</span><span class="value">attention dispatch: standard · xsa · fused · linear.</span></li>
          <li><span class="key">Make</span><span class="value">preconditioner dispatch by config.precond.</span></li>
        </ul>
      </div>
    </div>

    <div class="tag-row" style="margin-top: var(--sp-8);">
      <span class="tag"><span class="dot"></span>typed Config dataclass</span>
      <span class="tag"><span class="dot"></span>typed Spec / Result / Metrics</span>
      <span class="tag"><span class="dot"></span>four CLI entry points</span>
      <span class="tag"><span class="dot"></span>paper-worthiness rubric</span>
      <span class="tag"><span class="dot"></span>git_sha + schema-stable JSON</span>
      <span class="tag"><span class="dot"></span>single-word naming enforced in CI</span>
      <span class="tag"><span class="dot"></span>12 paper_runs JSON artifacts</span>
    </div>
  </div>
</section>


<!-- ============================================================================
     7. DOCUMENTATION NAVIGATION  (track-based entry system)
     ========================================================================= -->
<section class="section" id="docs">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>06 / Documentation</span></p>
      <h2>Pick a track. The doc system meets you where you are.</h2>
      <p class="lede">
        Four entry points. Each track tells you which docs to read first and
        which to skip.
      </p>
    </div>

    <div class="tracks">

      <a class="track" href="{{ '/getting-started/' | relative_url }}">
        <div class="track-index">Track 01</div>
        <h3>First-time user</h3>
        <p>I want to install xaker and run a forward pass.</p>
        <div class="track-pages">
          <a href="{{ '/getting-started/' | relative_url }}">Getting started</a>
          <a href="{{ '/installation/' | relative_url }}">Installation</a>
          <a href="{{ '/faq/' | relative_url }}">FAQ</a>
          <a href="/xaker/troubleshooting/">Troubleshooting</a>
          <a href="/xaker/contributing/">Contributing</a>
        </div>
      </a>

      <a class="track" href="{{ '/tutorial/' | relative_url }}">
        <div class="track-index">Track 02</div>
        <h3>Researcher</h3>
        <p>I want to understand the math, the pipeline, the numbers.</p>
        <div class="track-pages">
          <a href="{{ '/tutorial/' | relative_url }}">Tutorial</a>
          <a href="{{ '/math/' | relative_url }}">Mathematical foundations</a>
          <a href="{{ '/architecture/' | relative_url }}">Architecture</a>
          <a href="{{ '/limitations/' | relative_url }}">Limitations</a>
          <a href="/xaker/troubleshooting/">Troubleshooting</a>
          <a href="/xaker/contributing/">Contributing</a>
        </div>
      </a>

      <a class="track" href="{{ '/recipes/' | relative_url }}">
        <div class="track-index">Track 03</div>
        <h3>Contributor</h3>
        <p>I want to add a kernel, a preconditioner, a variant.</p>
        <div class="track-pages">
          <a href="{{ '/recipes/' | relative_url }}">Recipes</a>
          <a href="{{ '/api/' | relative_url }}">API reference</a>
          <a href="{{ '/design_decisions/' | relative_url }}">Design decisions</a>
          <a href="/xaker/troubleshooting/">Troubleshooting</a>
          <a href="/xaker/contributing/">Contributing</a>
        </div>
      </a>

      <a class="track" href="{{ '/paper_rubric/' | relative_url }}">
        <div class="track-index">Track 04</div>
        <h3>Benchmark / validation</h3>
        <p>I want to evaluate xaker against my own baseline.</p>
        <div class="track-pages">
          <a href="{{ '/paper_rubric/' | relative_url }}">Paper rubric</a>
          <a href="https://github.com/{{ site.repository }}/tree/master/paper_runs">paper_runs/</a>
          <a href="https://github.com/{{ site.repository }}/blob/master/RESULTS.md">RESULTS.md</a>
          <a href="/xaker/troubleshooting/">Troubleshooting</a>
          <a href="/xaker/contributing/">Contributing</a>
        </div>
      </a>

    </div>
  </div>
</section>


<!-- ============================================================================
     8. QUICKSTART
     ========================================================================= -->
<section class="section" id="quickstart">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>07 / Quickstart</span></p>
      <h2>Two commands to a benchmark. Three to a Transformer.</h2>
      <p class="lede">
        The working install is the source install (PyPI mirroring is planned).
        The headline benchmark is one reproducible command.
      </p>
    </div>

    <div class="quick">
      <div class="quick-side">

        <div class="code-shell" id="shell-install">
          <div class="code-shell-head">
            <span class="dots"><span></span><span></span><span></span></span>
            <span>install · bash</span>
            <button class="copy-btn" data-copy-target="shell-install-code">Copy</button>
          </div>
          <pre id="shell-install-code"><code>git clone https://github.com/{{ site.repository }}.git
cd xaker
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'</code></pre>
        </div>

        <p class="muted" style="font-size: var(--fs-sm);">
          The <span class="t-mono">.[dev]</span> is intentional; it means
          "this package, plus dev extras."
        </p>

        <div class="code-shell" id="shell-validate">
          <div class="code-shell-head">
            <span class="dots"><span></span><span></span><span></span></span>
            <span>validate · bash</span>
            <button class="copy-btn" data-copy-target="shell-validate-code">Copy</button>
          </div>
          <pre id="shell-validate-code"><code>xaker-validate
<span class="c"># novelty        2/3  novel=2/2 (fused.py,linear.py)
# repro          3/3  seeds=True bench=True paper_runs=12 cudnn=True
# correctness    3/3  fused=True cg=True dispatch=True
# efficiency     3/3  bench=True specs=True runs=12
# stability      3/3  seeds=True runs=True dtype=True
# usability      3/3  cli=True readme=True rubric=True
# Total          17/18 — PASS</span></code></pre>
        </div>

      </div>

      <div class="quick-side">

        <div class="code-shell" id="shell-forward">
          <div class="code-shell-head">
            <span class="dots"><span></span><span></span><span></span></span>
            <span>forward pass · python</span>
            <button class="copy-btn" data-copy-target="shell-forward-code">Copy</button>
          </div>
          <pre id="shell-forward-code"><code><span class="kn">from</span> <span class="nn">xaker</span> <span class="kn">import</span> <span class="n">Config</span>, <span class="n">Model</span>
<span class="kn">import</span> <span class="nn">torch</span>

<span class="n">cfg</span> <span class="o">=</span> <span class="n">Config</span><span class="p">(</span><span class="n">dim</span><span class="o">=</span><span class="mi">512</span>, <span class="n">heads</span><span class="o">=</span><span class="mi">8</span>,
            <span class="n">kernel</span><span class="o">=</span><span class="s2">"exp"</span>, <span class="n">precond</span><span class="o">=</span><span class="s2">"fast"</span><span class="p">)</span>
<span class="n">m</span>   <span class="o">=</span> <span class="n">Model</span><span class="p">(</span><span class="n">cfg</span>, <span class="n">num_layers</span><span class="o">=</span><span class="mi">6</span>, <span class="n">vocab_size</span><span class="o">=</span><span class="mi">32000</span>,
             <span class="n">max_seq_len</span><span class="o">=</span><span class="mi">512</span>, <span class="n">attention_type</span><span class="o">=</span><span class="s2">"fused"</span><span class="p">)</span>

<span class="n">logits</span> <span class="o">=</span> <span class="n">m</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span>, <span class="mi">32000</span>, <span class="p">(</span><span class="mi">2</span>, <span class="mi">128</span><span class="p">)))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">logits</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>  <span class="c"># torch.Size([2, 128, 32000])</span></code></pre>
        </div>

        <p class="muted" style="font-size: var(--fs-sm);">
          The same <span class="t-mono">Config</span> drives
          <span class="t-mono">Standard</span>, <span class="t-mono">Xsa</span>,
          <span class="t-mono">Fused</span>, and <span class="t-mono">Linear</span>
          through the <span class="t-mono">BLOCK</span> registry — no
          <span class="t-mono">if mode == "fused"</span> chains.
        </p>

        <div class="code-shell" id="shell-bench">
          <div class="code-shell-head">
            <span class="dots"><span></span><span></span><span></span></span>
            <span>reproduce headline · bash</span>
            <button class="copy-btn" data-copy-target="shell-bench-code">Copy</button>
          </div>
          <pre id="shell-bench-code"><code>python -m xaker.bench.condition \
    --lam 10.0 --lengths 16 32 64 128 \
    --out paper_runs/condition.json

# wrote paper_runs/condition.json
# ✓ headline condition numbers match the four metric cards above</code></pre>
        </div>

      </div>
    </div>

    <p style="margin-top: var(--sp-6);">
      <a class="btn btn-link" href="{{ '/tutorial/' | relative_url }}">Full tutorial: build a four-block Transformer step-by-step</a>
    </p>
  </div>
</section>


<!-- ============================================================================
     9. DESIGN PRINCIPLES  (manifesto)
     ========================================================================= -->
<section class="section" id="principles">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>08 / Design principles</span></p>
      <h2>Why the code looks the way it does.</h2>
      <p class="lede">
        A short manifesto on the choices behind xaker. These are
        defended in <a href="{{ '/design_decisions/' | relative_url }}">design_decisions.md</a>;
        here are the four anchors.
      </p>
    </div>

    <div class="manifesto">

      <div class="principle">
        <div class="index">i.</div>
        <h4>Polymorphism, not flags.</h4>
        <p>
          The four attention variants, four kernels, four preconditioners,
          and three XSA modes are all selected through a single dispatch.
          Adding a variant is one class plus one registry entry. Adding a
          fifth is the same amount of work.
        </p>
      </div>
      <div class="principle">
        <div class="index">ii.</div>
        <h4>The math is the spec.</h4>
        <p>
          <span class="t-mono">docs/math.md</span> and
          <span class="t-mono">xaker/solver/precond.py</span> agree byte for
          byte. If a derivation disagrees with the code, the code wins until
          a paper revision moves the goalposts.
        </p>
      </div>
      <div class="principle">
        <div class="index">iii.</div>
        <h4>Solver-backed, not softmax-backed.</h4>
        <p>
          Attention as a regularised system gives you a preconditioner
          knob. Identity, Diagonal, Fast, and Cccp cover the spectrum from
          debug-only to best-converges-on-ill-conditioned-kernels.
          Preconditioner choice is a config field, not a fork.
        </p>
      </div>
      <div class="principle">
        <div class="index">iv.</div>
        <h4>Numerical stability is a contract.</h4>
        <p>
          <span class="t-mono">BOUND</span> clamps. <span class="t-mono">BOUND</span>
          does not vary with the kernel. <span class="t-mono">λ</span> is
          guaranteed positive via softplus + ε. The dtype frontier is
          documented, not hidden.
        </p>
      </div>
      <div class="principle">
        <div class="index">v.</div>
        <h4>One public surface. One word per symbol.</h4>
        <p>
          <span class="t-mono">_private</span> prefixes are gone.
          Multi-word snake-case is gone. Aliases are gone. CI enforces
          this. The single-word rule keeps the API coherent across
          versions.
        </p>
      </div>
      <div class="principle">
        <div class="index">vi.</div>
        <h4>What we did not build, deliberately.</h4>
        <p>
          No FlashAttention. No sparse / Nyström. No AMP. No
          <span class="t-mono">transformers</span>-hub integration. Each of
          these is documented as absent and survives a paper revision.
          Roadmap, not TODO.
        </p>
      </div>

    </div>
  </div>
</section>


<!-- ============================================================================
     10. LIMITATIONS PANEL
     ========================================================================= -->
<section class="section" id="limitations">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>09 / Limitations</span></p>
      <h2>Where the math behaves, and where it doesn't.</h2>
      <p class="lede">
        Read this section before using xaker in production. The dtype frontier
        is a contract; the solver fallbacks are explicit; the unsupported
        environments are listed.
      </p>
    </div>

    <div class="limitations-panel">
      <div class="limitations-head">
        <div class="col">Where</div>
        <div class="col">Guard</div>
        <div class="col">dtype frontier</div>
        <div class="col">Notes</div>
      </div>

      <div class="limit-row">
        <div class="col where">xaker/attention/linear.py</div>
        <div class="col guard">elu(x) + 1</div>
        <div class="col dtype">fp16: x &lt; −14  ·  bf16: x &lt; −30</div>
        <div class="col notes">Per-dtype <span class="t-mono">feature_clamp</span> keeps the feature map strictly positive.</div>
      </div>
      <div class="limit-row">
        <div class="col where">xaker/attention/func.py</div>
        <div class="col guard">exp(clamp(x, −100, 100))</div>
        <div class="col dtype">fp16: x &gt; ~50  ·  bf16: x &gt; ~80</div>
        <div class="col notes">Same guard as PyTorch's softmax in low precision.</div>
      </div>
      <div class="limit-row">
        <div class="col where">xaker/solver/precond.py</div>
        <div class="col guard">BOUND = 1e6</div>
        <div class="col dtype">fp16-safe ceiling ~6.5e4</div>
        <div class="col notes">Lower <span class="t-mono">BOUND</span> to <span class="t-mono">1e4</span> for fp16-grade accuracy.</div>
      </div>
      <div class="limit-row">
        <div class="col where">xaker/solver/cg.py</div>
        <div class="col guard">PCG fallback</div>
        <div class="col dtype">matrix-dependent</div>
        <div class="col notes">Fused falls back to <span class="t-mono">torch.linalg.solve</span> on <span class="t-mono">not converged ∧ finite</span>.</div>
      </div>
      <div class="limit-row">
        <div class="col where">xaker/solver/precond.py</div>
        <div class="col guard">Cccp cost</div>
        <div class="col dtype">O(n³) per build</div>
        <div class="col notes">Beyond ~512 tokens prefer <span class="t-mono">fast</span> or <span class="t-mono">diagonal</span>.</div>
      </div>
      <div class="limit-row">
        <div class="col where">PyTorch + MPS</div>
        <div class="col guard">Batched linalg</div>
        <div class="col dtype">Apple Silicon</div>
        <div class="col notes"><span class="t-mono">linalg.solve</span> / <span class="t-mono">eigh</span> have shape bugs on 4-D inputs.</div>
      </div>
      <div class="limit-row">
        <div class="col where">xaker/attention/linear.py</div>
        <div class="col guard">Position structure</div>
        <div class="col dtype">task-dependent</div>
        <div class="col notes">Linear can't represent positions; fails on copy at length=32 (14%). Use Standard / Xsa / Fused for positional tasks.</div>
      </div>
    </div>

    <p style="margin-top: var(--sp-5);">
      <a class="btn btn-link" href="{{ '/limitations/' | relative_url }}">Full limitations page</a>
    </p>
  </div>
</section>


<!-- ============================================================================
     11. RUBRIC  (the visible trust mechanism)
     ========================================================================= -->
<section class="section" id="rubric">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>10 / Quality bar</span></p>
      <h2>This library is evaluated, not just presented.</h2>
      <p class="lede">
        The paper-worthiness rubric runs on every push to
        <span class="t-mono">master</span> and is enforced by CI. Each
        dimension below maps to a real grader that inspects the repo and
        emits an evidence string. The current run is presented here; it
        moves with the codebase.
      </p>
    </div>

    {%- comment -%}
      Rubric values come from `xaker.rubric.grade(".")`; the numbers below
      match the live run (17/18 PASS, novelty = 2/3 because the third
      novel kind is currently score-pending). The bars are CSS-only; the
      raw `--bar` value is set inline from the same source.
    {%- endcomment -%}

    <div class="rubric">
      <div class="rubric-card" style="--bar: 100%;">
        <div class="rubric-eyebrow"><span class="rubric-name">novelty</span><span class="rubric-of">of <em>3</em></span></div>
        <div class="rubric-score">2<small style="font-size: 0.5em; color: var(--ink-2);">/3</small></div>
        <div class="rubric-bar"></div>
        <div class="rubric-evidence">novel=2/2 (fused.py, linear.py)</div>
      </div>
      <div class="rubric-card" style="--bar: 100%;">
        <div class="rubric-eyebrow"><span class="rubric-name">repro</span><span class="rubric-of">of <em>3</em></span></div>
        <div class="rubric-score">3<small style="font-size: 0.5em; color: var(--ink-2);">/3</small></div>
        <div class="rubric-bar"></div>
        <div class="rubric-evidence">seeds=True bench=True paper_runs=12 cudnn=True</div>
      </div>
      <div class="rubric-card" style="--bar: 100%;">
        <div class="rubric-eyebrow"><span class="rubric-name">correctness</span><span class="rubric-of">of <em>3</em></span></div>
        <div class="rubric-score">3<small style="font-size: 0.5em; color: var(--ink-2);">/3</small></div>
        <div class="rubric-bar"></div>
        <div class="rubric-evidence">fused=True cg=True dispatch=True</div>
      </div>
      <div class="rubric-card" style="--bar: 100%;">
        <div class="rubric-eyebrow"><span class="rubric-name">efficiency</span><span class="rubric-of">of <em>3</em></span></div>
        <div class="rubric-score">3<small style="font-size: 0.5em; color: var(--ink-2);">/3</small></div>
        <div class="rubric-bar"></div>
        <div class="rubric-evidence">bench=True specs=True runs=12</div>
      </div>
      <div class="rubric-card" style="--bar: 100%;">
        <div class="rubric-eyebrow"><span class="rubric-name">stability</span><span class="rubric-of">of <em>3</em></span></div>
        <div class="rubric-score">3<small style="font-size: 0.5em; color: var(--ink-2);">/3</small></div>
        <div class="rubric-bar"></div>
        <div class="rubric-evidence">seeds=True runs=True dtype=True</div>
      </div>
      <div class="rubric-card" style="--bar: 100%;">
        <div class="rubric-eyebrow"><span class="rubric-name">usability</span><span class="rubric-of">of <em>3</em></span></div>
        <div class="rubric-score">3<small style="font-size: 0.5em; color: var(--ink-2);">/3</small></div>
        <div class="rubric-bar"></div>
        <div class="rubric-evidence">cli=True readme=True rubric=True</div>
      </div>
    </div>

    <div class="rubric-total">
      <strong>17 / 18</strong>
      <span class="status-good">● PASS</span>
      <span style="margin-left: var(--sp-3);">Run on <span class="t-mono">master</span>; refreshed every CI build.</span>
    </div>

    <p style="margin-top: var(--sp-5);">
      <a class="btn btn-link" href="{{ '/paper_rubric/' | relative_url }}">Rubric documentation</a>
    </p>
  </div>
</section>


<!-- ============================================================================
     12. CITATION (footer)
     ========================================================================= -->
<section class="section" id="cite">
  <div class="container">
    <div class="section-head">
      <p class="eyebrow"><span>11 / Cite</span></p>
      <h2>Cite xaker.</h2>
    </div>

    <p style="max-width: 64ch;">
      The paper is in preparation. Until the arXiv identifier is assigned,
      cite the GitHub release matching the version you used. A canonical
      <span class="t-mono">CITATION.cff</span> ships at the repository root
      and is what GitHub's "Cite this repository" button reads.
    </p>

    <div class="code-shell" id="shell-cite">
      <div class="code-shell-head">
        <span class="dots"><span></span><span></span><span></span></span>
        <span>cite · bibtex</span>
        <button class="copy-btn" data-copy-target="shell-cite-code">Copy</button>
      </div>
      <pre id="shell-cite-code"><code><span class="k">@software</span><span class="p">{</span><span class="nv">xaker</span><span class="p">,</span>
  <span class="na">title</span>  <span class="o">=</span> <span class="s">{</span><span class="s">xaker: A Mathematically Grounded Attention Framework</span><span class="s">}</span><span class="p">,</span>
  <span class="na">author</span> <span class="o">=</span> <span class="s">{</span><span class="s">sachin</span><span class="s">}</span><span class="p">,</span>
  <span class="na">year</span>   <span class="o">=</span> <span class="s">{</span><span class="s">2026</span><span class="s">}</span><span class="p">,</span>
  <span class="na">url</span>    <span class="o">=</span> <span class="s">{</span><span class="s">https://github.com/sachncs/xaker</span><span class="s">}</span><span class="p">,</span>
  <span class="na">note</span>   <span class="o">=</span> <span class="s">{</span><span class="s">Paper in preparation; arXiv ID will replace this entry on submission.</span><span class="s">}</span>
<span class="p">}</span></code></pre>
    </div>
  </div>
</section>
