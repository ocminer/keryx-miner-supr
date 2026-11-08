# License scope and reuse

The **Keryx Miner Supr Source-Sharing License 1.0** in [LICENSE](LICENSE) is the
license offered for **ocminer / Suprnova's original contributions** in this
revision. It is a custom source-available license, not an OSI-approved license.

- Running the original miner is permitted, including mining for profit.
- Incorporating our covered source or compiled kernels into a closed-source
  miner or another closed-source program requires separate written permission.
- Source-sharing derivatives must provide complete corresponding source,
  retain our credits, and link back to
  [ocminer/keryx-miner-supr](https://github.com/ocminer/keryx-miner-supr).
- These conditions apply to our copyrightable code and modifications, to the
  extent we own them. They do not claim ownership of upstream code, independent
  implementations, mathematical ideas, or third-party libraries and models.

## Which code is covered

The repository's commit history and existing notices identify the provenance
of each contribution. Our original additions and modifications to the CUDA,
OpenCL, Vulkan and Metal kernels, solver, scheduling, platform integration and
host code are covered to the extent owned by ocminer / Suprnova. A mixed file
can contain both our modifications and independently licensed upstream code;
this license does not replace the upstream license for the upstream portions.

Keryx Labs/Kaspa upstream code, vendored libraries, llama.cpp and patches from
other copyright holders, dependency crates, inference components with their
own license notices, and model weights retain their respective licenses.
The existing LICENSE-APACHE, LICENSE-MIT and THIRD_PARTY_NOTICES.txt files are
preserved as inherited notices, not as alternative licenses for newly covered
original contributions. Component-specific license files also remain in force.

## Earlier revisions

The first revision carrying this policy is included in tag **v0.16.3**.
Earlier commits, including the already-public optimization and build commits
`1079c2c` and `f27d088`, were published with the inherited license terms. Any
rights previously granted for those copies remain valid. This policy cannot
retroactively remove them or make previously authorized reuse unauthorized.
Future original contributions offered under this license use these new terms.

## Attribution example

Include this in your main README and distributed notices, with a clickable
link where supported; include it in an existing About/Credits/legal-notices
interface as well:

> Includes code and/or GPU kernels from keryx-miner-supr by ocminer / Suprnova.
> https://github.com/ocminer/keryx-miner-supr

Retain upstream credits too. The full conditions, scope and limitations are
in [LICENSE](LICENSE); the example does not replace that license.
