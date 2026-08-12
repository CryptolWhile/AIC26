import logging
import os
from typing import List, Dict, Any, Tuple
import shutil

logger = logging.getLogger(__name__)

class SubmitService:    
    def __init__(self):
        pass

    def submit(self, path: str, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        os.makedirs(path, exist_ok=True)
        for key, value in data.items():
            with open(os.path.join(path, f'{key}.csv'), 'w') as f:
                for item in value:
                    f.write(item + '\n')
        
        shutil.make_archive('submission', 'zip', path)

        return {'message': 'Submission successful'}, 200