from typing import List, Dict, Any  

class Reranking:
    def __init__(self):
        pass

    def rrf(self, list1: List[Dict[str, Any]], list2: List[Dict[str, Any]], 
            k: int = 60, id_field: str = 'image_id') -> List[Dict[str, Any]]:
        scores = {}     # Dictionary lưu tổng điểm RRF của từng item (Key: item_id, Value: tổng điểm)
        item_data = {}  # Dictionary lưu nội dung chi tiết của item để sau này tái tạo lại kết quả
        
        # --- BƯỚC 1: Xử lý danh sách thứ nhất (list1) ---
        for rank, item in enumerate(list1, start=1):
            # Lấy mã định danh của item (ví dụ: 'id' của khung hình)
            item_id = item.get(id_field)
            # Nếu item không có trường id, dùng toàn bộ nội dung item ép thành chuỗi làm id thay thế
            if item_id is None:
                item_id = str(item)
            
            # Tính điểm RRF dựa trên thứ hạng. Hằng số k mặc định là 60.
            score = 1.0 / (k + rank)
            
            # Nếu item này lần đầu tiên xuất hiện, khởi tạo điểm = 0 và copy dữ liệu vào kho lưu trữ
            if item_id not in scores:
                scores[item_id] = 0
                item_data[item_id] = item.copy()
            
            # Cộng dồn điểm RRF
            scores[item_id] += score
            # Lưu lại lịch sử thứ hạng của item này ở list1
            if 'ranks' not in item_data[item_id]:
                item_data[item_id]['ranks'] = {}
            item_data[item_id]['ranks']['list1'] = rank
        
        # --- BƯỚC 2: Xử lý danh sách thứ hai (list2) ---
        for rank, item in enumerate(list2, start=1):
            item_id = item.get(id_field)
            if item_id is None:
                item_id = str(item)
            
            score = 1.0 / (k + rank)
            
            if item_id not in scores:
                scores[item_id] = 0
                item_data[item_id] = item.copy()
                if 'ranks' not in item_data[item_id]:
                    item_data[item_id]['ranks'] = {}
            
            scores[item_id] += score
            item_data[item_id]['ranks']['list2'] = rank
        
        # --- BƯỚC 3: Tổng hợp và Sắp xếp lại kết quả ---
        reranked_results = []
        # Sắp xếp các item theo tổng điểm (total_score) từ cao xuống thấp 
        for item_id, total_score in sorted(scores.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True):
            result = item_data[item_id].copy()
            result['score'] = total_score  
            reranked_results.append(result)
        
        # danh sách kết quả đã được sắp xếp theo điểm RRF
        return sorted(reranked_results, key=lambda x: x['score'], reverse=True)

    def rrf_multiple(self, *lists: List[Dict[str, Any]], 
                       k: int = 60, id_field: str = 'image_id') -> List[Dict[str, Any]]:

        if len(lists) == 0:
            return []
        
        elif len(lists) == 1:
            return lists[0]
        
        # Lấy danh sách đầu tiên làm mốc khởi đầu
        result = lists[0]
        
        # Duyệt qua các danh sách còn lại từ vị trí thứ 1 đến hết
        for i in range(1, len(lists)):
            cleaned_result = []
            for item in result:
                clean_item = item.copy()
                # Xóa sạch điểm số và lịch sử rank cũ để tính toán công bằng cho lần gộp tiếp theo
                clean_item.pop('score', None)
                clean_item.pop('ranks', None)
                cleaned_result.append(clean_item)
            
            # Gọi hàm rrf gộp danh sách hiện tại với danh sách tiếp theo trong mảng `lists`
            result = self.rrf(cleaned_result, lists[i], k=k, id_field=id_field)
        
        # Trả về kết quả cuối cùng sắp xếp theo điểm giảm dần
        return sorted(result, key=lambda x: x['score'], reverse=True)

    def rrf_services(self, list:List[Dict[str, Any]], k:int = 60, id_field:str = 'image_id') -> List[Dict[str, Any]]:

        # Semantic dùng Milvus (metric_type="L2"). L2 distance càng NHỎ càng tốt -> phải xếp tăng dần (reverse=False)
        semantic_list = sorted(list, key=lambda x: x.get('semantic', float('inf')), reverse=False)

        # OCR/ASR thường là độ chính xác (confidence), càng LỚN càng tốt -> xếp giảm dần (reverse=True)
        ocr_list = sorted(list, key=lambda x: x.get('ocr', 0), reverse=True)
        # asr_list = sorted(list, key=lambda x: x.get('asr', 0), reverse=True)

        # Gộp 2 danh sách qua thuật toán rrf_multiple
        return self.rrf_multiple(semantic_list, 
                                ocr_list, 
                                k=k, 
                                id_field=id_field)

    def weighted_sum(self, list:List[Dict[str, Any]], weights:List[float]) -> List[Dict[str, Any]]:
        for item in list:
            # Semantic là L2 distance (0->2), chuyển ngược lại: lấy số âm hoặc (2 - distance) để distance càng nhỏ điểm càng cao.
            # Ở đây em dùng số âm (-semantic) để đồng bộ với logic "điểm càng to càng tốt".
            semantic_score = -item.get('semantic', float('inf'))
            
            # Tính tổng điểm
            score = semantic_score * weights[0] + item.get('asr', 0) * weights[1] + item.get('ocr', 0) * weights[2]
            item['score'] = score  

        return sorted(list, key=lambda x: x.get('score', float('-inf')), reverse=True)

    def combined_reranking(self, 
            list:List[Dict[str, Any]],
            weights:List[float]=[0.5, 0.3, 0.2],
            k:int = 60, 
            id_field:str = 'image_id') -> List[Dict[str, Any]]:

        rrf_result = self.rrf_services(list, k=k, id_field=id_field)
        # (Mặc định: 50% semantic, 30% asr, 20% ocr)
        weighted_result = self.weighted_sum(list, weights=weights)

        return self.rrf_multiple(rrf_result, weighted_result, k=k, id_field=id_field)