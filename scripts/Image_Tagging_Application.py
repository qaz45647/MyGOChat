import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import csv

class ModernImageTagger:
    def __init__(self, master):
        self.master = master
        master.title("圖片標籤工具")
        master.configure(bg='#f0f0f0')
        master.geometry("1200x900")  # 增加視窗大小

        # 標籤集合
        self.tag_categories = {
            '情緒': {'快樂', '難過', '無奈', '不耐煩', '厭世', '感傷', 
                    '緊張', '興奮', '憤怒', '沮喪', '懷疑', '期待', '驚訝', '困惑', '孤獨', '害怕', '幸福', '感恩', '平淡'},
            '態度': {'輕蔑', '挑釁', '諷刺', '冷漠', '防禦', '自嘲', 
                    '不屑', '傲慢', '懷疑', '堅定', '妥協', '逃避', '投降', '強硬', '調侃', '尊敬', '樂觀', '謙遜', '友善', '好奇', '輕浮', '慶幸', '失望', '壓抑', '隱忍', '認真'},
            '互動': {'質疑', '道歉', '懇求', '確認', '拒絕', '勸說', 
                    '挑釁', '安慰', '邀請', '命令', '詢問', '嘗試說服', '批評', '求助', '調解', '否定', '肯定', '反思', '放棄', '讓步', '釋懷', '讚美', '打招呼', '說明感受', '吐槽'},
            '語調': {'口語', '直接', '感性', '理性', '諷刺', '幽默', 
                    '嚴肅', '輕鬆', '含蓄', '激動', '冷靜', '真誠', '挖苦', '鼓舞', '欲言又止', '控訴', '不甘','哀傷'}
        }
        
        # 標籤類別對應的英文名稱
        self.category_names = {
            '情緒': 'EMOTION_TAGS',
            '態度': 'ATTITUDE_TAGS',
            '互動': 'INTERACTION_TAGS',
            '語調': 'TONE_TAGS'
        }
        
        # 初始化變數
        self.image_paths = self.load_images_from_folder('IMG')
        self.current_index = 0
        self.selected_tags = {category: set() for category in self.tag_categories}
        self.tag_buttons = {}
        
        # 已標記的圖片集合
        self.tagged_images = self.load_previous_tags()
        # 創建 CSV 檔案（如果不存在）
        self.create_csv()
        # 創建UI
        self.create_widgets()

    def load_previous_tags(self):
        tagged_images = {}
        if os.path.exists('result.csv'):
            try:
                df = pd.read_csv('result.csv', encoding='utf-8-sig')
                for _, row in df.iterrows():
                    image_name = row['Image']
                    tagged_images[image_name] = {}
                    
                    for category in self.tag_categories:
                        column_name = self.category_names[category]
                        tags_str = str(row.get(column_name, ''))
                        
                        if tags_str and tags_str != 'nan':
                            tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                            tagged_images[image_name][category] = set(tags)
                        else:
                            tagged_images[image_name][category] = set()
            except Exception as e:
                messagebox.showwarning("讀取錯誤", f"無法讀取 result.csv: {str(e)}")
        
        return tagged_images

    def create_csv(self):
        if not os.path.exists('result.csv'):
            with open('result.csv', 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                header = ['Image', 'Image_NoExt']
                for category in self.tag_categories:
                    header.append(self.category_names[category])
                writer.writerow(header)

    def save_current_image_tags(self):
        # 寫入當前圖片的標籤到 CSV
        current_image = os.path.basename(self.image_paths[self.current_index])
        
        # 更新已標記圖片的標籤
        current_tags = {category: set(tags) for category, tags in self.selected_tags.items()}
        self.tagged_images[current_image] = current_tags

        # 只保存有標籤的圖片
        rows = []
        for img_name, img_tags in self.tagged_images.items():
            # 檢查是否有任何非空的標籤
            if any(len(tags) > 0 for tags in img_tags.values()):
                # 取得不含副檔名的檔案名稱
                img_name_no_ext = os.path.splitext(img_name)[0]
                row = [img_name, img_name_no_ext]
                for category in self.tag_categories:
                    row.append(','.join(img_tags.get(category, [])))
                rows.append(row)

        # 寫入 CSV
        with open('result.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            header = ['Image', 'Image_NoExt']
            for category in self.tag_categories:
                header.append(self.category_names[category])
            writer.writerow(header)
            writer.writerows(rows)

    def load_images_from_folder(self, folder_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            messagebox.showinfo("提示", f"已創建 {folder_path} 資料夾，請放入圖片")
            return []
        
        image_paths = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f)) and 
               os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if not image_paths:
            messagebox.showinfo("提示", f"{folder_path} 資料夾中沒有圖片")
        return sorted(image_paths)  # 排序圖片路徑

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 圖片顯示區域
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(pady=10, fill=tk.X)
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(expand=True)

        # 進度顯示和跳轉框架
        progress_frame = ttk.Frame(image_frame)
        progress_frame.pack(pady=5)

        # 進度顯示
        self.progress_label = ttk.Label(progress_frame, text="", font=("Arial", 12))
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # 跳轉輸入框和按鈕
        jump_label = ttk.Label(progress_frame, text="跳轉到第", font=("Arial", 12))
        jump_label.pack(side=tk.LEFT, padx=5)
        
        self.jump_entry = ttk.Entry(progress_frame, width=5)
        self.jump_entry.pack(side=tk.LEFT, padx=2)
        
        jump_label_end = ttk.Label(progress_frame, text="張圖片", font=("Arial", 12))
        jump_label_end.pack(side=tk.LEFT, padx=2)
        
        jump_button = ttk.Button(progress_frame, text="跳轉", command=self.jump_to_image)
        jump_button.pack(side=tk.LEFT, padx=5)

        # 使用 Notebook 作為標籤分類
        self.tag_notebook = ttk.Notebook(main_frame)
        self.tag_notebook.pack(pady=10, fill=tk.BOTH, expand=True)

        for category, tags in self.tag_categories.items():
            # 創建標籤框架
            tag_frame = ttk.Frame(self.tag_notebook)
            self.tag_notebook.add(tag_frame, text=category)
            
            # 創建標籤按鈕框架
            tag_buttons_frame = ttk.Frame(tag_frame)
            tag_buttons_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            
            # 使用網格佈局
            self.create_tag_buttons_grid(tag_buttons_frame, tags, category)

        # 導航按鈕框架
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(pady=10)

        # 上一張和下一張圖按鈕
        prev_button = ttk.Button(nav_frame, text="上一張圖", command=self.prev_image)
        prev_button.pack(side=tk.LEFT, padx=5)
        next_button = ttk.Button(nav_frame, text="下一張圖", command=self.next_image)
        next_button.pack(side=tk.LEFT, padx=5)

        if self.image_paths:
            self.show_image()

    def jump_to_image(self):
        if not self.image_paths:
            return
        
        try:
            target_index = int(self.jump_entry.get()) - 1  # 轉換為從0開始的索引
            if 0 <= target_index < len(self.image_paths):
                self.save_current_image_tags()
                self.current_index = target_index
                self.show_image()
                self.jump_entry.delete(0, tk.END)  # 清空輸入框
            else:
                messagebox.showwarning("警告", f"請輸入1到{len(self.image_paths)}之間的數字")
        except ValueError:
            messagebox.showwarning("警告", "請輸入有效的數字")

    def create_tag_buttons_grid(self, parent, tags, category):
        self.tag_buttons[category] = {}
        sorted_tags = sorted(tags)
        
        # 計算網格大小
        columns = 5  # 固定 5 列
        for i, tag in enumerate(sorted_tags):
            row = i // columns
            col = i % columns
            btn = ttk.Button(
                parent, 
                text=tag, 
                width=10,
                command=lambda t=tag, c=category: self.toggle_tag(c, t)
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky='ew')
            self.tag_buttons[category][tag] = btn

        # 配置所有列等寬
        for col in range(columns):
            parent.grid_columnconfigure(col, weight=1)

    def show_image(self):
        if not self.image_paths:
            return
            
        image = Image.open(self.image_paths[self.current_index])
        image.thumbnail((800, 500))
        photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=photo)
        self.image_label.image = photo
        
        # 更新進度顯示
        self.progress_label.config(text=f"進度: {self.current_index + 1} / {len(self.image_paths)}")
        
        # 切換到情緒標籤頁
        self.tag_notebook.select(0)  # 選擇第一個標籤頁（情緒）
        
        # 重置所有標籤按鈕顏色和選擇狀態
        for category in self.tag_buttons:
            self.selected_tags[category].clear()
            for tag, btn in self.tag_buttons[category].items():
                btn.state(['!pressed'])

        # 載入之前的標籤
        current_image_name = os.path.basename(self.image_paths[self.current_index])
        
        if current_image_name in self.tagged_images:
            for category, tags in self.tagged_images[current_image_name].items():
                for tag in tags:
                    if tag in self.tag_buttons[category]:
                        btn = self.tag_buttons[category][tag]
                        btn.state(['pressed'])
                        self.selected_tags[category].add(tag)

    def toggle_tag(self, category, tag):
        btn = self.tag_buttons[category][tag]
        
        if tag in self.selected_tags[category]:
            self.selected_tags[category].remove(tag)
            btn.state(['!pressed'])
        else:
            self.selected_tags[category].add(tag)
            btn.state(['pressed'])

    def next_image(self):
        self.save_current_image_tags()
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.show_image()
        # 如果所有圖片已標記完
        if self.current_index == 0:
            messagebox.showinfo("完成", "所有圖片已標記完成")

    def prev_image(self):
        self.save_current_image_tags()
        self.current_index = (self.current_index - 1 + len(self.image_paths)) % len(self.image_paths)
        self.show_image()

def main():
    root = tk.Tk()
    app = ModernImageTagger(root)
    root.mainloop()

if __name__ == "__main__":
    main()