from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .transformer_vfe import TransformerVFE

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'TransformerVFE': TransformerVFE
}
