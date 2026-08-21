from pickle import LIST
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from datetime import datetime
import faiss
from langdetect import detect

from src.services.embedding.service import EmbeddingService
from src.services.llm.service import ChatModelService
from src.core import extensions
from src.models.keyframe_model import Keyframe
from src.models.video_model import Video
from src.core.config import Settings
from src.services.retrieval.translator import Translator
from src.services.retrieval.temporal import TemporalSolver  
from src.services.retrieval.reranking import Reranking

logger = logging.getLogger(__name__)

class RetrievalService:
    """Service for handling video search and retrieval operations."""
    
    def __init__(self):
        """Initialize the retrieval service with required dependencies."""
        try:
            self.embedding_service = EmbeddingService()
            
            # Initialize LLM service 
            self.llm_service = ChatModelService()
            gemini_key = Settings().gemini.GEMINI_API_KEY
            if gemini_key:
                self.llm_service.register_model(
                    model_name="gemini-1.5-flash",
                    provider_name="gemini",
                    config={
                        "api_key": gemini_key
                    }
                )

            self.translator = Translator()

            self.reranking = Reranking()

            logger.info("RetrievalService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RetrievalService: {str(e)}")
            raise

    # @property
    # def faiss_db(self):
    #     """Get FAISS connection dynamically."""
    #     return extensions.faiss_db

    @property
    def milvus_db(self):
        """Get Milvus connection dynamically."""
        return extensions.milvus_db
    
    @property
    def elastic_db(self):
        """Get ElasticSearch connection dynamically."""
        return extensions.elastic_db
    
    @property
    def metadata_store_db(self):
        """Get Metadata Store connection dynamically."""
        return extensions.metadata_store_db if hasattr(extensions, 'metadata_store_db') else None

    def get_datasets(self) -> List[str]:
        """
        Get list of available datasets.
        
        Returns:
            List of dataset identifiers
        """
        try:
            return ["All", "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"]
        except Exception as e:
            logger.error(f"Error in get_datasets: {str(e)}")
            raise

    def get_videos(self, dataset: Optional[str] = None) -> List[str]:
        """
        Get list of available videos for a dataset.
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            List of video identifiers
        """
        try:
            # Mock: Mock video data - replace with actual database query
            videos_by_dataset = {
                "All": ["All", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08"],
                "D01": ["All", "V04", "V05", "V06"],
                "D02": ["All", "V01", "V02", "V03"], 
                "D03": ["All", "V01", "V02", "V03"],
                "D04": ["All", "V01", "V02", "V03"],
                "D05": ["All", "V01", "V02", "V03"],
                "D06": ["All", "V01", "V02", "V03"],
                "D07": ["All", "V01", "V02", "V03", "V04"],
                "D08": ["All", "V08"]
            }
            
            return videos_by_dataset.get(dataset or "All", [])
            
        except Exception as e:
            logger.error(f"Error in get_videos: {str(e)}")
            raise

    def search_videos(
        self,
        prompt: str,
        extra_prompts: Optional[List[str]] = None,
        ocr_search: Optional[str] = None,
        dataset_filter: Optional[str] = None,
        video_filter: Optional[str] = None,
        limit: int = 5,
        models: Optional[List[str]] = None,
        rerank_method: Optional[str] = "All"
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-modal video search.
        
        Args:
            prompt: Main query
            extra_prompts: Extra prompts for temporal search
            ocr_search: Text to search for in OCR (optional)
            dataset_filter: Dataset to filter by (optional)
            video_filter: Video to filter by (optional)
            limit: Maximum number of results
            models: Models to use for search
            
        Returns:
            List of formatted search results
        """
        try:
            logger.info(f"Processing search: prompt='{prompt}', extra_prompts='{extra_prompts}', ocr_search='{ocr_search}', models='{models}', dataset_filter='{dataset_filter}', video_filter='{video_filter}', limit='{limit}'")
            all_results = []
            for model in models:
                # Perform the multi-modal search
                logger.info(f"Processing model {model}")
                config = self.embedding_service.create_config(model)
                self.embedding_service.registry_model(
                    config=config["config"],
                    model_name=config['model_name'],
                    provider_name=config['provider_name']
                )
                search_results = self._perform_multi_modal_search(
                    prompt=prompt,
                    extra_prompts=extra_prompts,
                    ocr_search=ocr_search,
                    # asr_search=asr_search,
                    dataset_filter=dataset_filter,
                    video_filter=video_filter,
                    limit=limit,
                    model={"collection": model, 
                            "model_name": config["model_name"]},
                    rerank_method=rerank_method
                )
                all_results.append(search_results)
            final_results = self._union_all_models(all_results, limit, rerank_method)
            return final_results
            
        except Exception as e:
            logger.error(f"Error in search_videos: {str(e)}")
            raise


    def _semantic_search(self, query: str, 
                        model: Dict[str, Any],
                        filters: Tuple[str, str],
                        limit: int = 800) -> List[Dict[str, Any]]:
        """Perform semantic search using embedding models."""
        try:
            try:
                if detect(query) == 'vi':
                    query = self.translator(query)
            except Exception as e:
                logger.warning(f"Language detection failed: {e}, using original query")

            query_embedding = self.embedding_service.embed_text(
                            texts=[query],
                            model_name=model["model_name"],
                            batch_size=32,
                            normalize=True
            )
            # if self.faiss_db is None:
            #     logger.error(f"FAISS database connection is None for model {model['collection']}")
            #     return []
            #     
            # results = self.faiss_db.search(
            #     collection_name=model["collection"],
            #     query={
            #         "embedding": query_embedding.embeddings[0],
            #         "k": limit,
            #         "metric_type": faiss.METRIC_INNER_PRODUCT,
            #         "hnsw": 200
            #     }
            # )
            # 
            # _results = []
            # for result in results:
            #     if filters[0] and filters[1]:
            #         request_dataset, request_video = filters
            #         curr_dataset, curr_video = result['metadata']['id'].split('_')[0], result['metadata']['id'].split('_')[1]
            #         if request_dataset != 'All' and request_dataset != curr_dataset:
            #             continue
            #         if request_video != 'All' and request_video != curr_video:
            #             continue
            #     _results.append({
            #         "image_id": result['metadata']['id'],
            #         "semantic": result['distance'],
            #         "timestamp": result['metadata']['timestamp']
            #     })

            if self.milvus_db is None:
                logger.error(f"Milvus database connection is None for model {model['collection']}")
                return []
                
            query_vector = query_embedding.embeddings[0].tolist()
            
            # Query Milvus
            results = self.milvus_db.search(
                collection_name=model["collection"],
                query={
                    "data": [query_vector],
                    "anns_field": "embedding",
                    "param": {
                        "metric_type": "L2",
                        "params": {"nprobe": 10}
                    },
                    "limit": limit,
                    "output_fields": ["id", "metadata"]
                }
            )
            
            _results = []
            # results[0] contains the hits for our single query vector
            if results and len(results) > 0 and len(results[0]) > 0:
                for hit in results[0]:
                    if filters[0] and filters[1]:
                        request_dataset, request_video = filters
                        curr_dataset, curr_video = hit.id.split('_')[0], hit.id.split('_')[1]
                        if request_dataset != 'All' and request_dataset != curr_dataset:
                            continue
                        if request_video != 'All' and request_video != curr_video:
                            continue
                    
                    # Extract timestamp from metadata JSON string
                    try:
                        import json
                        metadata_dict = json.loads(hit.entity.get("metadata", "{}"))
                        timestamp = metadata_dict.get("timestamp", 0)
                    except Exception:
                        timestamp = 0
                        
                    _results.append({
                        "image_id": hit.id,
                        "semantic": hit.distance,
                        "timestamp": timestamp
                    })

            if not results:
                logger.info(f"No results found for query in collection {model['collection']}")
                return []
            
            return _results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}, model: {model}, query: {query}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _perform_multi_modal_search(
        self,
        prompt: str,
        extra_prompts: List[str],
        ocr_search: str,
        dataset_filter: str,
        video_filter: str,
        rerank_method: str,
        limit: int,
        model: str
    ) -> List[Dict[str, Any]]:
        """Perform multi-modal search."""
        try:
            if extra_prompts:
                all_prompts = [prompt] + extra_prompts
                all_results = []
                for prompt in all_prompts:
                    single_semantic_search = self._semantic_search(
                        query=prompt, 
                        model=model,
                        filters=(dataset_filter, video_filter) 
                    )
                    logger.info(f"Single semantic search: {len(single_semantic_search)} {single_semantic_search[:2]}")
                    ocr_results = self._ocr_search(
                        ocr_search=ocr_search, 
                        filters=(dataset_filter, video_filter) 
                    )
                    logger.info(f"Single OCR search: {len(ocr_results)} {ocr_results[:2]}")
                    # asr_results = self._asr_search(
                    #     asr_search=asr_search,
                    #     filters=(dataset_filter, video_filter)
                    # )
                    formatted_search_results = self._format_search_results(
                        single_semantic_search, 
                        ocr_results
                        # asr_results
                    )
                    logger.info(f"Formatted search results: {len(formatted_search_results)} {formatted_search_results[:2]}")
                    if rerank_method == "rrf":
                        single_search_results = self.reranking.rrf_services(
                            formatted_search_results
                        )
                    elif rerank_method == "weighted_sum":
                        single_search_results = self.reranking.weighted_sum(
                            formatted_search_results,
                            weights=[1.0, 0.0, 0.0]
                        )
                    else:
                        single_search_results = self.reranking.combined_reranking(
                            formatted_search_results
                        )
                    logger.info(f"Single search results: {len(single_search_results)} {single_search_results[:2]}")
                    all_results.append(single_search_results)
                _results = self._temporal_search(
                    input=all_results, 
                    limit=limit
                )
                logger.info(f"Temporal search results: {len(_results)} {_results[:2]}")
            else:
                semantic_results = self._semantic_search(
                    query=prompt, 
                    model=model,
                    filters=(dataset_filter, video_filter) 
                )
                ocr_results = self._ocr_search(
                    ocr_search=ocr_search, 
                    filters=(dataset_filter, video_filter) 
                )
                # ASR search
                # asr_search = self._asr_search(
                #     asr_search=asr_search,
                #     filters=(dataset_filter, video_filter)
                # )
                formatted_search_results = self._format_search_results(
                    semantic_results, 
                    ocr_results
                    # asr_results
                )
                if rerank_method == "rrf":
                    _results = self.reranking.rrf_services(
                        formatted_search_results
                    )
                elif rerank_method == "weighted_sum":
                    _results = self.reranking.weighted_sum(
                        formatted_search_results,
                        weights=[1.0, 0.0, 0.0]
                    )
                else:
                    _results = self.reranking.combined_reranking(
                        formatted_search_results
                    )
            return _results
            
        except Exception as e:
            logger.error(f"Error in perform_multi_modal_search: {str(e)}")
            return []
        

    def _temporal_search(self, input: List[List[Dict[str, Any]]], 
                         limit: int) -> List[Dict[str, Any]]:
        """Perform temporal search for scene transitions."""
        try:
            temporal_solver = TemporalSolver(
                lists=input,
                top_k=limit
            )
            _results = temporal_solver.solve()
            return _results
            
        except Exception as e:
            logger.error(f"Error in temporal search: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _ocr_search(self, ocr_search: str, filters: Tuple[str, str]) -> List[Dict[str, Any]]:
        """Perform OCR-based text search."""

        try:
            if ocr_search == "":
                return []
            dataset_filter, video_filter = filters
            
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "ocr": {
                                        "query": ocr_search,
                                        "operator": "OR",
                                        "minimum_should_match": "10%",
                                        "fuzziness": "AUTO"
                                    }
                                }
                            }
                        ],
                        "filter": []
                    }
                },
                "size": 800
            }

            if dataset_filter != "All":
                query["query"]["bool"]["filter"].append({
                    "term": {"dataset_id": dataset_filter}
                })
            if video_filter != "All":
                query["query"]["bool"]["filter"].append({
                    "term": {"video_id": video_filter}
                })

            results = self.elastic_db.client.search(
                index="vieocr",
                body=query
            )

            formatted_results = []
            for hit in results["hits"]["hits"]:
                source = hit["_source"]
                formatted_results.append({
                    "image_id": source.get("frame_id"),
                    "ocr": hit["_score"]
                })

            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in OCR search: {str(e)}")
            return []

    def _union_all_models(self, results: List[List[Dict[str, Any]]], limit: int, rerank_method: str) -> List[List[Dict[str, Any]]]:
        """Combine and rank search results from different models."""
        try:
            # Lọc bỏ các kết quả rỗng (ví dụ model H chưa có dữ liệu)
            non_empty = [r for r in results if r]
            if not non_empty:
                return [[]]

            # Nếu chỉ có 1 model có kết quả, trả về luôn
            if len(non_empty) == 1:
                final_res = non_empty[0]
            else:
                # Nếu có nhiều hơn 1 model, kiểm tra xem đây là kết quả temporal hay bình thường
                is_temporal = non_empty and len(non_empty[0]) > 0 and isinstance(non_empty[0][0], list)
                if is_temporal:
                    # Gộp kết quả temporal (nối mảng) vì RRF không hỗ trợ list of lists
                    merged = []
                    for model_res in non_empty:
                        merged.extend(model_res)
                    final_res = merged
                else:
                    # Gộp chúng bằng rrf_multiple cho kết quả bình thường
                    final_res = self.reranking.rrf_multiple(*non_empty)
                
            # Đảm bảo kết quả cuối cùng được sắp xếp
            is_temporal = final_res and isinstance(final_res[0], list)
            if is_temporal:
                # Tính tổng điểm của một sequence temporal
                def get_temporal_score(seq):
                    return sum(event.get('score', 0) for event in seq) if isinstance(seq, list) else 0
                final_res = sorted(final_res, key=get_temporal_score, reverse=True)
            else:
                final_res = sorted(final_res, key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0, reverse=True)
            
            return [final_res[:limit]]
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [[]]

    def _format_search_results(self, *results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format search results for frontend consumption."""
        def mapping(list:List[Dict[str, Any]]) -> Dict[str, Any]:
            result = {}
            for item in list:
                result[item['image_id']] = item
            return result
        semantic_dict = mapping(results[0])
        ocr_dict = mapping(results[1])
        # asr_dict = mapping(results[2])
        all_keys = set(semantic_dict.keys()) | set(ocr_dict.keys()) # | set(asr_dict.keys())
        formatted_dict = {}
        for key in all_keys:
            semantic = semantic_dict[key]['semantic'] if key in semantic_dict else 0
            ocr = ocr_dict[key]['ocr'] if key in ocr_dict else 0
            
            timestamp = 0
            if key in semantic_dict and 'timestamp' in semantic_dict[key]:
                timestamp = semantic_dict[key]['timestamp']
            elif key in ocr_dict:
                key_info = self.metadata_store_db.search(
                    collection_name="keyframe",
                    query={
                        "id": key
                    }
                )
                timestamp = key_info['timestamp']

            formatted_dict[key] = {
                "image_id": key,
                "semantic": semantic,
                "ocr": ocr,
                # "asr": asr
                "timestamp": timestamp
            }
        # for key, value in asr_dict.items():
        #     formatted_dict[key]['asr'] = value.get('asr', 0)

        formatted = []
        for key, value in formatted_dict.items():
            formatted.append(value)
        
        return formatted

