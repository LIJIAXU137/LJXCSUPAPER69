CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds  
MM '25, October 27-31, 2025, Dublin, Ireland.
A **Proof of Theorem 1**
PROOF. At step $t-1$, we have
$$\hat {\mathbf {\Phi }}_{t-1}=\left (\mathbf {E}_{1:t-2}^{\top }\mathbf {E}_{1:t-2}+\mathbf {E}_{t-1}^{\top }\mathbf {E}_{t-1}+\gamma \mathbf {I}\right )^{-1}\left [\mathbf {E}_{1:t-1}^{\top }\mathbf {Y}_{1:t-2}^{\text {train}}+\mathbf {E}_{t-1}^{\top }\bar {\mathbf {Y}}_{t-1}^{\text {train}}\quad \mathbf {E}_{t-1}^{\top }\tilde {\mathbf {Y}}_{t-1}^{\text {train}}\right ].$$
Hence, at step $t$, we have
$$\hat {\bm {\Phi }}_{t}=\left (\mathbf {E}_{1:t-1}^{\top }\mathbf {E}_{1:t-1}+\mathbf {E}_{t}^{\top }\mathbf {E}_{t}+\gamma \mathbf {I}\right )^{-1}\left [\mathbf {E}_{1:t-1}^{\top }\mathbf {Y}_{1:t-1}^{\text {train}}+\mathbf {E}_{t}^{\top }\bar {\mathbf {Y}}_{t}^{\text {train}}\quad \mathbf {E}_{t}^{\top }\tilde {\mathbf {Y}}_{t}^{\text {train}}\right ].$$
We have defined the inverted auto-correlation matrix $\Psi_{t-1}$ in the paper via
$$\Psi _{t-1}=(\mathbf {E}_{1:t-2}^{\top }\mathbf {E}_{1:t-2}+\mathbf {E}_{t-1}^{\top }\mathbf {E}_{t-1}+\gamma \mathbf {I})^{-1}.$$
To facilitate subsequent calculations, here we also define a cross-correlation matrix $Q_{t-1}$, i.e.,
Thus we can rewrite Eqn (15) as
$$\mathbf {Q}_{t-1}=\left [\mathbf {E}_{1:t-1}^{\top }\mathbf {Y}_{1:t-2}^{\text {train}}+\mathbf {E}_{t-1}^{\top }\bar {\mathbf {Y}}_{t-1}^{\text {train}}\quad \mathbf {E}_{t-1}^{\top }\tilde {\mathbf {Y}}_{t}^{\text {train}}\right ].$$
$$\hat {\Phi }_{t-1}=\Psi _{t-1}\mathrm {Q}_{t-1}.$$
Therefore, at step $t$ we have
$$\hat {\Phi }_{t}=\Psi _{t}\mathrm {Q}_{t}.$$
From Eqn (17), we can recursively calculate $\Psi_t$ from $\Psi_{t-1}$, i.e.,
$$\Psi _{t}=(\Psi _{t-1}^{-1}+\mathbf {E}_{t}^{\top }\mathbf {E}_{t})^{-1}.$$
According to the Woodbury matrix identity, we have
$$(\mathbf {A}+\mathbf {U}\mathbf {C}\mathbf {V})^{-1}=\mathbf {A}^{-1}-\mathbf {A}^{-1}\mathbf {U}(\mathbf {C}^{-1}+\mathbf {V}\mathbf {A}^{-1}\mathbf {U})^{-1}\mathbf {V}\mathbf {A}^{-1}.$$
Let $A = \Psi_{t-1}^{-1}$, $U = E_t^\top$, $C = I$, and $V = E_t$ in Eqn (21), we have
$$\Psi _{t}=\Psi _{t-1}-\Psi _{t-1}\mathbf {E}_{t}^{\top }(\mathbf {I}+\mathbf {E}_{t}\mathbf {\Psi }_{t-1}\mathbf {E}_{t}^{\top })^{-1}\mathbf {E}_{t}\mathbf {\Psi }_{t-1}.$$
Hence, $\Psi_t$ can be recursively updated using its last-phase counterpart $\Psi_{t-1}$ and data from the current phase (i.e., $E_t$). This proves the recursive calculation of the inverted auto-correlation matrix.
Next, we derive the recursive formulation of $\hat{\Phi}_t$. To this end, we also recourse the cross-correlation matrix $Q_t$ at step $t$, i.e.,
$$\mathbf {Q}_{t}=\begin{bmatrix}\mathbf {E}_{1:t-1}^{\top }\mathbf {Y}_{1:t-1}^{\text {train}}+\mathbf {E}_{t}^{\top }\bar {\mathbf {Y}}_{t}^{\text {train}}&\mathbf {E}_{t}^{\top }\bar {\mathbf {Y}}_{t}^{\text {train}}\end{bmatrix}=\mathbf {Q}_{t-1}^{\prime }+\begin{bmatrix}\mathbf {E}_{t}^{\top }\bar {\mathbf {Y}}_{t}^{\text {train}}&\mathbf {E}_{t}^{\top }\bar {\mathbf {Y}}_{t}^{\text {train}}\end{bmatrix},$$
where
$$\mathrm {Q}_{t-1}^{\prime }=\left \{\begin{array}{ll}\left [\mathrm {Q}_{t-1}\quad \mathbf {0}_{d_{\mathrm {E}}\times \left ( d_{C_{t}}-d_{C_{t-1}}\right )}\right ],& d_{C_{t}}>d_{C_{t-1}}\\
 \mathrm {Q}_{t-1},& d_{C_{t}}=d_{C_{t-1}}\end{array}\right ..$$
Note that the concatenation in Eqn (25) is due to the assumption that $Y_{1:t}^{\text{train}}$ at step $t$ contains more data classes (hence more columns) than $Y_{1:t-1}^{\text{train}}$. It is possible that there are no new classes appear at step $t$, then $\tilde{Y}_t^{\text{train}}$ should be 0.
Similar to what Eqn (25) does,
$$\hat {\Phi }_{(t-1)}^{\prime }=\begin{cases}\begin{bmatrix}\hat {\Phi }_{t-1}&\mathbf {0}_{d_{\mathrm {E}}\times (d_{C_{t}}-d_{C_{t-1}})}\end{bmatrix},&d_{C_{t}}>d_{C_{t-1}}\\
 \hat {\Phi }_{t-1},&d_{C_{t}}=d_{C_{t-1}}\end{cases}$$
We have
$$\hat {\Phi }_{(t-1)},=\Psi _{t-1}\mathbf {Q}_{t-1}^{\prime }.$$
Hence, $\hat{\Phi}_t$ can be rewritten as
$$\begin{array}{l}{\hat {\Phi }_{t}=\Psi _{t}\mathrm {Q}_{t}}\\
{=\Psi _{t}(\mathrm {Q}_{t-1}^{\prime }+\left [\mathrm {E}_{t}^{\top }\mathrm {\tilde {Y}}_{t}^{\mathrm {train}}\quad \mathrm {E}_{t}^{\top }\mathrm {\tilde {Y}}_{t}^{\mathrm {train}}\right ])}\\
{=\Psi _{t}\mathrm {Q}_{t-1}^{\prime }+\Psi _{t}\mathrm {E}_{t}^{\top }\left [\mathrm {\tilde {Y}}_{t}^{\mathrm {train}\quad \mathrm {\tilde {Y}}_{t}^{\mathrm {train}}}\right ].}\\
\end{array}$$
By substituting Eqn (23) into $\Psi_t Q_{t-1}'$, we have
$$\begin{array}{c}\Psi _{t}\mathbf {Q}_{t-1}^{\prime }=\Psi _{t-1}\mathbf {Q}_{t-1}^{\prime }-\Psi _{t-1}\mathbf {E}_{t}^{\top }(\mathbf {I}+\mathbf {E}_{t}\mathbf {\Psi }_{t-1}\mathbf {E}_{t}^{\top })^{-1}\mathbf {E}_{t}\mathbf {\Psi }_{t-1}\mathbf {Q}_{t-1}^{\prime }\\
=\mathbf {\hat {\Phi }}_{(t-1)},-\Psi _{t-1}\mathbf {E}_{t}^{\top }(\mathbf {I}+\mathbf {E}_{t}\mathbf {\Psi }_{t-1}\mathbf {E}_{t}^{\top })^{-1}\mathbf {E}_{t}\mathbf {\hat {\Phi }}_{(t-1)},.\end{array}$$
(15)
(16)
(17)
(18)
(19)
(20)
(21)
(22)
(23)
(24)
(25)
(26)
(27)
(28)
(29)
MM '25, October 27–31, 2025, Dublin, Ireland. Jiaxu Li et al.
To simplify this equation, let $K_t = (I + E_t \Psi_{t-1} E_t^\top)^{-1}$. Since
$$\mathbf {I}=\mathbf {K}_{t}\mathbf {K}_{t}^{-1}=\mathbf {K}_{t}(\mathbf {I}+\mathbf {E}_{t}\mathbf {\Psi }_{t-1}\mathbf {E}_{t}^{\top }),$$
We have $K_t = I - K_t E_t \Psi_{t-1} E_t^\top$. Therefore,
$$\begin{array}{l}\boldsymbol {\Psi }_{t-1}\mathbf {E}_{t}^{\top }\left (\mathbf {I}+\mathbf {E}_{t}\boldsymbol {\Psi }_{t-1}\mathbf {E}_{t}^{\top }\right )^{-1}\\
=\boldsymbol {\Psi }_{t-1}\mathbf {E}_{t}^{\top }\mathbf {K}_{t}\\
=\boldsymbol {\Psi }_{t-1}\mathbf {E}_{t}^{\top }\left (\mathbf {I}-\mathbf {K}_{t}\mathbf {E}_{t}\boldsymbol {\Psi }_{t-1}\mathbf {E}_{t}^{\top }\right )\\
=(\boldsymbol {\Psi }_{t-1}-\boldsymbol {\Psi }_{t-1}\mathbf {E}_{t}^{\top }\mathbf {K}_{t}\mathbf {E}_{t}\boldsymbol {\Psi }_{t-1})\mathbf {E}_{t}^{\top }\\
=\boldsymbol {\Psi }_{t}\mathbf {E}_{t}^{\top }.\end{array}$$
Substituting Eqn (30) into Eqn (29), $\Psi_t Q_{t-1}'$ can be written as
$$\Psi _{t}\mathbf {Q}_{t-1}^{\prime }=\mathbf {\hat {\Phi }}_{(t-1)},-\mathbf {\Psi }_{t}\mathbf {E}_{t}^{\top }\mathbf {E}_{t}\mathbf {\hat {\Psi }}_{(t-1)}.$$
Substituting Eqn (31) into Eqn (28) implies that
$$\begin{array}{c}\hat {\Phi }_{t}=\hat {\Phi }_{(t-1)},-\mathbf {\Psi }_{t}\mathbf {E}_{t}^{\top }\mathbf {E}_{t}\hat {\Phi }_{(t-1)},+\mathbf {\Psi }_{t}\mathbf {E}_{t}^{\top }\left [\mathbf {\tilde {Y}}_{t}^{\text {train}}\quad \mathbf {\tilde {Y}}_{t}^{\text {train}}\right ]\\
=\left [\mathbf {\hat {\Phi }}_{t-1}-\mathbf {\Psi }_{t}\mathbf {E}_{t}^{\top }\mathbf {E}_{t}\hat {\Phi }_{t-1}+\mathbf {\Psi }_{t}\mathbf {E}_{t}^{\top }\mathbf {\tilde {Y}}_{t}^{\text {train}}\quad \mathbf {\Psi }_{t}\mathbf {E}_{t}^{\top }\mathbf {\tilde {Y}}_{t}^{\text {train}}\right ]\end{array}$$
which completes the proof.
(30)
(31)
(32)
□