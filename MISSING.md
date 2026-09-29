# Files still needed before this compiles

Two things the manuscript depends on are not present in the working folder and
must be added by hand (they are probably on Overleaf, not on this machine):

## 1. Bibliography — `cleaned_ref.bib`

`manuscript.tex` ends with `\bibliography{cleaned_ref}`, but no `.bib` file
exists anywhere in the project folder. Without it, LaTeX emits every citation
as `?`.

## 2. Figures — four PDFs referenced by `manuscript.tex`

    Fig/Pipeline_figure.pdf     Figure 1  — analysis pipeline
    Fig/Panel 1.pdf             Figure 2  — spatial organisation / robustness
    Fig/Panel 2.pdf             Figure 3  — cross-patient / cross-platform validation
    Fig/Panel 3.pdf             Figure S1 — autoencoder vulnerability analysis

`Fig/` currently holds three differently-named PNGs (`LOPO_validation.png`,
`PermutationControl.png`, `VisiumHD_validation.png`) which the manuscript does
not reference.

Note the two filenames containing a space; either keep them exactly as written
or rename both the files and the `\includegraphics` calls.
