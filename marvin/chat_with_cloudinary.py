from marvin.beta.assistants import Assistant
import cloudinary
import cloudinary.api
import pydantic

class Usage(pydantic.BaseModel):
    rpt_date: str    
    transformations: int
    storage: int
    bandwidth: int

    def __init__(self, rpt_date: str, transformations: int, storage: int, bandwidth: int):
        super().__init__()
        self.rpt_date = rpt_date
        self.transformations=transformations,
        self.storage=storage,
        self.bandwidth=bandwidth

    def __str__(self):
        return f"Usage: {self.transformations} transformations, {self.storage} storage, {self.bandwidth} bandwidth"

def get_cloudinary_usage(rpt_date: str)->Usage:
    """ Fetch the usage on Cloudinary for the date provided. """
    resp = cloudinary.api.usage(date=rpt_date)
    print(resp)
    """Fetch Cloudinary usage data"""
    return Usage(
        rpt_date = rpt_date,
        transformations=resp.get('transformations').get('usage'),
        storage=resp.get('bandwidth').get('usage'),
        bandwidth=resp.get('storage').get('usage')
    )

#ai = Assistant(tools=[get_cloudinary_usage])
#ai.say("Can you provide me the usage on \"2024-10-01\"?")

data = {'plan': 'CSMs Rock', 'last_updated': '2024-11-18', 'date_requested': '2024-10-01T00:00:00Z', 'transformations': {'usage': 10, 'breakdown': {'transformation': 10}}, 'objects': {'usage': 3215}, 'bandwidth': {'usage': 13672261}, 'storage': {'usage': 6601490235}, 'requests': 25, 'resources': 767, 'derived_resources': 2448, 'seconds_delivered': None, 'media_limits': {'image_max_size_bytes': 314572800, 'video_max_size_bytes': 1048576000, 'raw_max_size_bytes': 1048576000, 'image_max_px': 50000000, 'asset_max_total_px': 50000000}}
print(data['transformations']['usage'])