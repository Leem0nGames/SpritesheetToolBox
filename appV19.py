# Imports
import sys
import os
import traceback
import math
import json
from PyQt6.QtGui import (QDragEnterEvent, QDropEvent, QPixmap, QIcon, QImage,
                         QPainter, QColor, QPen, QAction, QFont)
from PyQt6.QtCore import (QUrl, QThread, pyqtSignal, Qt, QSize, pyqtSlot, QPoint, QTimer)
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QLineEdit, QFileDialog, QRadioButton, QSpinBox, QComboBox,
                             QGroupBox, QTextEdit, QMessageBox, QSpacerItem, QSizePolicy,
                             QListWidget, QListWidgetItem, QGridLayout, QAbstractItemView,
                             QButtonGroup, QCheckBox, QProgressBar, QStyle, QSlider, QMenu)
from PIL import Image, ImageOps, ImageQt
import rembg
import cv2
import numpy as np

# --- Función para cargar config ---
def load_config(filepath="config.json"):
    default_config = {
        "app_info": { "window_title": "Sprite Extractor Pro", "logo_filename": "app_logo.png", "max_logo_width": 180, "stylesheet_path": "styles/dark_theme.qss" },
        "defaults": { "output_dir": "output_sprites", "preview_size": 128, "engine": "Rembg (Default)", "min_area": 100, "resampling": "LANCZOS", "pixelate_size": 4, "quantize_colors": 64, "quantize_method": "MEDIANCUT", "regen_layout_mode": "Columns", "regen_columns": 8, "regen_rows": 8, "regen_spacing": 2, "thumbnail_size": 96, "resize_output_enabled": False, "resize_output_width": 64, "resize_output_height": 64, "resize_output_resampling": "LANCZOS" },
        "options": { "engines": ["Rembg (Default)", "Fine-Tuned (Placeholder)"], "resampling_methods": ["LANCZOS", "NEAREST", "BILINEAR", "BICUBIC"], "quantize_methods": ["MEDIANCUT", "MAXCOVERAGE", "FASTOCTREE"], "preview_sizes": [128, 256, 512, 1024], "spinbox_ranges": { "custom_size": [16, 4096], "min_area": [10, 10000], "pixelate_size": [2, 32], "quantize_colors": [2, 256], "regen_columns": [1, 100], "regen_rows": [1, 100], "regen_spacing": [0, 100], "resize_output_width": [8, 4096], "resize_output_height": [8, 4096] }, "slider_ranges": {"thumbnail_size": [48, 256, 32]} },
        "ui_text": { "input_placeholder": "Selecciona o arrastra imagen(es)...", "input_button": "Entrada...", "output_placeholder": "Directorio salida...", "output_button": "Salida...", "files_group": "1. Archivos", "extract_group": "2. Opciones Extracción", "engine_label": "Motor IA:", "preview_size_label": "Tamaño Canvas Trabajo:", "custom_checkbox": "Personalizado", "min_area_label": "Área Mín.:", "resampling_label": "Remuestreo:", "extract_button": "Extraer Sprites", "effects_group": "3. Efectos (Opcional)", "pixelate_checkbox": "Pixelizar", "pixelate_size_label": "Tamaño Píxel:", "quantize_checkbox": "Reducir Colores", "quantize_colors_label": "Máx Colores:", "quantize_method_label": "Método:", "apply_effects_button": "Aplicar Efectos", "revert_effects_button": "Revertir Originales", "preview_group": "4. Vista Previa", "thumbnail_size_label": "Zoom Thumbs:", "load_button": "Cargar...", "save_selected_button": "Guardar Sel...", "delete_button": "Eliminar Sel.", "sprite_count_label": "Sprites: {count}", "output_options_group": "5. Opciones de Salida", "resize_output_group": "Redimensionar Salida", "resize_output_checkbox": "Redimensionar al guardar/generar", "resize_output_width_label": "Ancho Final:", "resize_output_height_label": "Alto Final:", "resize_output_resampling_label": "Remuestreo:", "regen_layout_label": "Priorizar:", "regen_layout_cols": "Columnas", "regen_layout_rows": "Filas", "regen_columns_label": "Columnas:", "regen_rows_label": "Filas:", "regen_spacing_label": "Espacio:", "regen_button": "Crear Spritesheet", "log_group": "Registro", "confirm_delete_title": "Confirmar", "confirm_delete_text": "¿Eliminar {num} sprite(s)?" }
    }
    try:
        with open(filepath, 'r', encoding='utf-8') as f: config = json.load(f)
        def update_recursive(d, u):
            for k, v in u.items():
                if isinstance(v, dict): current_val = d.get(k); d[k] = update_recursive(current_val, v) if isinstance(current_val, dict) else json.loads(json.dumps(v))
                elif k not in d: d[k] = v
            return d
        config = update_recursive(config, default_config)
        return config
    except FileNotFoundError:
        print(f"Info: Config '{filepath}' no hallado. Usando/creando defaults.")
        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception as e: print(f"Error: No se pudo guardar config default: {e}")
        return default_config
    except (json.JSONDecodeError, Exception) as e:
        print(f"ADVERTENCIA: Error cargando config '{filepath}' ({e}). Usando defaults.")
        return json.loads(json.dumps(default_config))

# --- Función para cargar hoja de estilos ---
def load_stylesheet_from_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f: return f.read()
    except FileNotFoundError: print(f"ADVERTENCIA: Stylesheet no hallado: '{filepath}'"); return "QWidget{background-color:#f0f0f0;color:#000;}"
    except Exception as e: print(f"ERROR cargando stylesheet '{filepath}': {e}"); return "QWidget{background-color:#f0f0f0;color:#000;}"

# --- Clases de Motores ---
class BaseExtractionEngine:
    def __init__(self, **kwargs): pass
    def process(self, img_in: Image.Image) -> Image.Image: raise NotImplementedError

class RembgEngine(BaseExtractionEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.session = None
        try: self.session = rembg.new_session()
        except Exception as e: print(f"Error inicializando sesión Rembg: {e}")
    def process(self, img_in: Image.Image) -> Image.Image:
        if not self.session: print("Error: Sesión Rembg no inicializada."); return img_in.copy()
        img_proc = img_in.copy()
        if img_proc.mode not in ['RGBA', 'RGB']:
            try: img_proc = img_proc.convert('RGB').convert('RGBA') if img_proc.mode == 'P' else img_proc.convert('RGBA')
            except Exception as e: print(f"Error conv a RGBA pre-Rembg: {e}"); return img_in.copy()
        try:
            img_out = rembg.remove(img_proc, session=self.session)
            if img_out is None: print(f"Error: rembg.remove devolvió None."); return img_proc
            if img_out.mode != 'RGBA': print(f"Warn: Salida Rembg no RGBA ({img_out.mode}). Convirtiendo."); img_out = img_out.convert('RGB').convert('RGBA') if img_out.mode == 'P' else img_out.convert('RGBA')
            return img_out
        except Exception as e: print(f"Error rembg.remove: {e}\n{traceback.format_exc()}"); return img_proc

class PlaceholderFineTunedEngine(BaseExtractionEngine):
    def __init__(self, model_path="?", **kwargs): super().__init__(**kwargs); self.model_path = model_path
    def process(self, img_in: Image.Image) -> Image.Image:
        print(f"[PlaceholderEngine] Modelo '{self.model_path}'..."); img_out = img_in.copy()
        try:
            if img_out.mode != 'RGBA': img_out = img_out.convert('RGB').convert('RGBA') if img_out.mode == 'P' else img_out.convert('RGBA')
            r, g, b, a = img_out.split(); r = ImageOps.invert(r); g = ImageOps.invert(g); b = ImageOps.invert(b); img_out = Image.merge('RGBA', (r, g, b, a))
        except Exception as e:
            print(f"Error PlaceholderEngine: {e}"); temp_orig = img_in.copy()
            if temp_orig.mode != 'RGBA':
                try: temp_orig = temp_orig.convert('RGB').convert('RGBA') if temp_orig.mode == 'P' else temp_orig.convert('RGBA')
                except Exception: return img_in.copy()
            return temp_orig
        return img_out

# --- Workers ---
class ExtractionWorker(QThread):
    progress = pyqtSignal(str); thumbnail_ready = pyqtSignal(object); extraction_data_ready = pyqtSignal(list)
    finished = pyqtSignal(str); error = pyqtSignal(str); progress_update = pyqtSignal(int)
    def __init__(self, engine: BaseExtractionEngine, input_path, min_area, target_canvas_size, resampling_filter, thumbnail_size_tuple):
        super().__init__(); self.engine=engine; self.input_path=input_path; self.min_area=min_area; self.target_canvas_size=target_canvas_size; self.resampling_filter=resampling_filter; self.thumbnail_pil_size=thumbnail_size_tuple; self._is_running=True; self._extracted_canvases=[]; self._error_occurred=False
    def stop(self): self._is_running = False; self.progress.emit("Deteniendo Extracción...")
    def run(self):
        self.progress.emit(f"Worker.run() para: {os.path.basename(self.input_path)}");
        if not self._is_running: self.progress.emit("Worker detenido pre-inicio."); return
        fig_count=0; final_msg="OK."; self._error_occurred=False; num_contours=-1
        try:
            self.progress.emit(f"P1: Abriendo..."); img_orig = Image.open(self.input_path);
            if not self._is_running: return
            self.progress.emit(f"P2: Abierto (Modo: {img_orig.mode}). P3: Procesando IA...");
            if not self.engine: self.error.emit("Motor no prop."); self._error_occurred=True; return
            img_bg = self.engine.process(img_orig);
            self.progress.emit(f"P4: IA OK (Modo: {img_bg.mode if img_bg else 'N/A'})");
            if not self._is_running: return
            if img_bg is None: self.error.emit("Motor devolvió None."); self._error_occurred=True; return
            if img_bg.mode != 'RGBA':
                try: img_bg = img_bg.convert('RGB').convert('RGBA') if img_bg.mode == 'P' else img_bg.convert('RGBA')
                except Exception as e: self.error.emit(f"Err conv post-IA a RGBA: {e}"); self._error_occurred=True; return
            self.progress.emit(f"P5: Imagen RGBA.")
            cv_img = np.array(img_bg)
            if cv_img.size == 0: final_msg = "Imagen vacía post-IA."; num_contours=0
            else:
                if len(cv_img.shape)<3 or cv_img.shape[2]<4: self.error.emit("No Alpha post-IA."); self._error_occurred=True; return
                alpha = cv_img[:, :, 3]; self.progress.emit("Buscando contornos...");
                contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); num_contours = len(contours)
                self.progress.emit(f"{num_contours} contornos."); self.progress_update.emit(0)
                for i, contour in enumerate(contours):
                    if not self._is_running: return
                    if num_contours > 0: self.progress_update.emit(int((i + 1) * 100 / num_contours))
                    if cv2.contourArea(contour) < self.min_area: continue
                    x,y,w,h=cv2.boundingRect(contour);
                    if w<=0 or h<=0: continue
                    x,y=max(0,x),max(0,y); x2,y2=min(x+w,img_bg.width),min(y+h,img_bg.height); cw,ch=x2-x,y2-y
                    if cw<=0 or ch<=0: continue
                    try: crop=img_bg.crop((x,y,x2,y2))
                    except IndexError as e: self.progress.emit(f"Adv: Crop err {i}: {e}."); continue
                    if crop.width<=0 or crop.height<=0: continue
                    fig_count+=1; canvas=Image.new('RGBA',(self.target_canvas_size,self.target_canvas_size),(0,0,0,0)); cr_w,cr_h=crop.size
                    m=0.95; s=min((self.target_canvas_size*m)/cr_w,(self.target_canvas_size*m)/cr_h); s=max(0.001,s); nw,nh=int(cr_w*s),int(cr_h*s); nw,nh=max(1,nw),max(1,nh)
                    try: rz_spr = crop.resize((nw,nh), resample=self.resampling_filter)
                    except Exception as e: self.progress.emit(f"Adv: Resize err {fig_count}: {e}."); continue
                    px,py=(self.target_canvas_size-nw)//2,(self.target_canvas_size-nh)//2
                    if rz_spr.mode != 'RGBA':
                        try: rz_spr = rz_spr.convert('RGB').convert('RGBA') if rz_spr.mode == 'P' else rz_spr.convert('RGBA')
                        except Exception as e: self.progress.emit(f"Adv: Err conv sprite {fig_count}: {e}."); continue
                    canvas.paste(rz_spr, (px, py), rz_spr); thumb = canvas
                    try:
                        if self.thumbnail_pil_size[0]>0 and self.thumbnail_pil_size[1]>0: thumb = canvas.resize(self.thumbnail_pil_size, Image.Resampling.LANCZOS)
                    except Exception as e: self.progress.emit(f"Adv: Thumb err {fig_count}: {e}."); thumb = canvas.copy()
                    self._extracted_canvases.append(canvas.copy()); self.thumbnail_ready.emit(thumb.copy())
            if cv_img.size > 0 or num_contours == 0:
                if num_contours >= 0 : self.progress_update.emit(100)
                if self._extracted_canvases: self.extraction_data_ready.emit(self._extracted_canvases)
                total_extracted = len(self._extracted_canvases)
                if fig_count == 0 and num_contours > 0: final_msg = f"OK ({os.path.basename(self.input_path)}). No figs > {self.min_area}px²."
                elif fig_count == 0 and num_contours == 0: final_msg = f"OK ({os.path.basename(self.input_path)}). No figs halladas."
                elif total_extracted == 0: final_msg = f"OK ({os.path.basename(self.input_path)}). Figs detectadas, 0 sprites finales."
                else: final_msg = f"OK ({os.path.basename(self.input_path)}): {total_extracted} sprites."
        except FileNotFoundError: self.error.emit(f"Err: Archivo no hallado: {self.input_path}"); self._error_occurred = True; return
        except Exception as e: self.error.emit(f"Err Extr ({type(e).__name__}) '{os.path.basename(self.input_path)}': {e}"); self._error_occurred = True; return
        finally: self._is_running = False; self.progress_update.emit(100) if not self._error_occurred and num_contours >= 0 else None
        if not self._error_occurred: self.finished.emit(final_msg)

class SpritesheetWorker(QThread):
    progress=pyqtSignal(str); finished=pyqtSignal(str); error=pyqtSignal(str); progress_update=pyqtSignal(int)
    def __init__(self, sprites, layout_mode, columns, rows, spacing, output_path):
        super().__init__(); self.sprites=[s.copy() for s in sprites if isinstance(s,Image.Image)]; self.layout_mode=layout_mode; self.columns=max(1,columns); self.rows=max(1,rows); self.spacing=max(0,spacing); self.output_path=output_path; self._is_running=True; self._error_occurred=False
    def stop(self): self._is_running = False; self.progress.emit("Deteniendo Regen...")
    def run(self):
        final_msg=""; num_s=0; self._error_occurred=False
        try:
            num_s=len(self.sprites);
            if num_s == 0: self.finished.emit("Sheet: No sprites."); return
            spr_w, spr_h = self.sprites[0].size
            if spr_w <= 0 or spr_h <= 0: self.error.emit("Tamaño sprite inválido."); self._error_occurred=True; return
            if self.layout_mode=='Columns': sh_cols=self.columns; sh_rows=math.ceil(num_s/sh_cols) if sh_cols>0 else num_s
            elif self.layout_mode=='Rows': sh_rows=self.rows; sh_cols=math.ceil(num_s/sh_rows) if sh_rows>0 else num_s
            else: sh_cols=self.columns; sh_rows=math.ceil(num_s/sh_cols) if sh_cols>0 else num_s; self.progress.emit("Warn: Modo layout inválido, usando cols.")
            self.progress.emit(f"Layout: {sh_cols}c x {sh_rows}f.")
            sh_w=max(1,(sh_cols*spr_w)+max(0,(sh_cols-1))*self.spacing); sh_h=max(1,(sh_rows*spr_h)+max(0,(sh_rows-1))*self.spacing)
            self.progress.emit(f"Creando sheet ({sh_w}x{sh_h})..."); sheet=Image.new('RGBA',(sh_w,sh_h),(0,0,0,0)); self.progress_update.emit(0)
            for i, spr in enumerate(self.sprites):
                if not self._is_running: self.progress.emit("Sheet cancelado."); return
                if num_s > 0: self.progress_update.emit(int((i+1)*100/num_s))
                r,c=i//sh_cols, i%sh_cols; px,py=c*(spr_w+self.spacing),r*(spr_h+self.spacing)
                if spr and spr.size==(spr_w,spr_h):
                    p_spr = spr
                    if p_spr.mode!='RGBA':
                        try: p_spr = p_spr.convert('RGB').convert('RGBA') if p_spr.mode=='P' else p_spr.convert('RGBA')
                        except Exception as e: self.progress.emit(f"Warn: Err conv sprite {i}: {e}. Skip."); continue
                    sheet.paste(p_spr, (px,py), p_spr)
                else: sz=f"{spr.size}" if spr else "N/A"; self.progress.emit(f"Warn: Skip sprite {i} (size: {sz} vs {spr_w}x{spr_h}).")
            if not self._is_running: return
            fpath=self.output_path; supp=(".png",".webp",".tiff")
            if not fpath.lower().endswith(supp): base,_=os.path.splitext(fpath); fpath=base+".png"; self.progress.emit(f"Adv: Guardando PNG: {fpath}")
            odir=os.path.dirname(fpath);
            if odir: os.makedirs(odir, exist_ok=True)
            self.progress.emit(f"Guardando sheet: {fpath}"); sheet.save(fpath); final_msg=f"Sheet guardado:\n{os.path.abspath(fpath)}"
        except Exception as e: self.error.emit(f"Err gen sheet:\n{traceback.format_exc()}"); self._error_occurred=True; return
        finally: self._is_running = False; self.progress_update.emit(100 if not self._error_occurred and num_s > 0 else 0)
        if final_msg and not self._error_occurred: self.finished.emit(final_msg)

class EffectsWorker(QThread):
    effect_applied=pyqtSignal(int, object); progress=pyqtSignal(str); finished=pyqtSignal(str); error=pyqtSignal(str); progress_update=pyqtSignal(int)
    def __init__(self, original_sprites, params): super().__init__(); self.orig= [s.copy() for s in original_sprites if isinstance(s,Image.Image)]; self.params=params; self._is_running=True; self._error_occurred=False
    def stop(self): self._is_running=False; self.progress.emit("Deteniendo Efectos...")
    def run(self):
        final_msg=""; total_s=0; self._error_occurred=False
        try:
            total_s=len(self.orig);
            if total_s == 0: self.finished.emit("No sprites."); return
            self.progress.emit(f"Aplicando a {total_s} sprites..."); px_en, px_sz = self.params.get('pixelate',False), self.params.get('pixel_size',4); qz_en, qz_col, qz_m_s = self.params.get('quantize',False), self.params.get('quantize_colors',64), self.params.get('quantize_method','MEDIANCUT')
            q_map={'MEDIANCUT': Image.Quantize.MEDIANCUT, 'MAXCOVERAGE': Image.Quantize.MAXCOVERAGE, 'FASTOCTREE': Image.Quantize.FASTOCTREE}; qz_m=q_map.get(qz_m_s.upper(), Image.Quantize.MEDIANCUT); self.progress_update.emit(0)
            for i, orig_ref in enumerate(self.orig):
                if not self._is_running: self.progress.emit("Efectos cancelados."); return
                self.progress_update.emit(int((i+1)*100/total_s)); img=orig_ref.copy(); w,h=img.size
                if img.mode!='RGBA':
                    try: img = img.convert('RGB').convert('RGBA') if img.mode == 'P' else img.convert('RGBA')
                    except Exception as e: self.progress.emit(f"Warn: Err conv sprite {i}: {e}. Skip."); self.effect_applied.emit(i, orig_ref.copy()); continue
                if px_en and px_sz>1 and w>=px_sz and h>=px_sz:
                    try: sw, sh = max(1, w//px_sz), max(1, h//px_sz); small = img.resize((sw, sh), Image.Resampling.NEAREST); img = small.resize((w, h), Image.Resampling.NEAREST)
                    except Exception as e: self.progress.emit(f"Warn: Pixelate err {i}: {e}")
                if qz_en and qz_col>1:
                    try: alpha = img.getchannel('A'); img_rgb = img.convert('RGB'); q_p = img_rgb.quantize(colors=qz_col, method=qz_m, dither=Image.Dither.FLOYDSTEINBERG); q_rgba = q_p.convert('RGBA'); q_rgba.putalpha(alpha); img = q_rgba
                    except Exception as e: self.progress.emit(f"Warn: Quantize err {i}: {e}")
                self.effect_applied.emit(i, img.copy())
            if not self._is_running: return; final_msg="Efectos aplicados."
        except Exception as e: self.error.emit(f"Error efectos:\n{traceback.format_exc()}"); self._error_occurred=True; return
        finally: self._is_running=False; self.progress_update.emit(100 if not self._error_occurred and total_s>0 else 0)
        if final_msg and not self._error_occurred: self.finished.emit(final_msg)

class ReorderableListWidget(QListWidget):
    items_reordered=pyqtSignal()
    def __init__(self, parent=None): super().__init__(parent); self.setDragEnabled(True); self.setAcceptDrops(True); self.setDropIndicatorShown(True); self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove); self.setDefaultDropAction(Qt.DropAction.MoveAction); self.setMovement(QListWidget.Movement.Snap); self.setFlow(QListWidget.Flow.LeftToRight); self.setWrapping(True); self._draw_grid=True
    def setGridVisible(self, visible):
        if self._draw_grid != visible: self._draw_grid = visible; self.viewport().update()
    def dropEvent(self, event: QDropEvent):
        source = event.source()
        if source == self and event.dropAction() == Qt.DropAction.MoveAction: super().dropEvent(event); self.items_reordered.emit(); event.acceptProposedAction()
        else: event.ignore()
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._draw_grid and self.viewMode() == QListWidget.ViewMode.IconMode:
            painter = QPainter(self.viewport()); pen = QPen(QColor(60, 60, 60)); pen.setStyle(Qt.PenStyle.DotLine); pen.setWidth(1); painter.setPen(pen)
            try:
                gs = self.gridSize(); sx, sy = gs.width(), gs.height()
                if sx <= 0 or sy <= 0: return
                vr = self.viewport().rect(); scr_x, scr_y = self.horizontalScrollBar().value(), self.verticalScrollBar().value()
                start_x = sx * math.floor(scr_x / sx); start_y = sy * math.floor(scr_y / sy)
                cx = start_x + sx;
                while cx < scr_x + vr.width() + sx: painter.drawLine(int(cx - scr_x), vr.top(), int(cx - scr_x), vr.bottom()); cx += sx
                cy = start_y + sy;
                while cy < scr_y + vr.height() + sy: painter.drawLine(vr.left(), int(cy - scr_y), vr.right(), int(cy - scr_y)); cy += sy
            except Exception as e: print(f"Grid paint err: {e}")
            finally:
                if painter and painter.isActive(): painter.end()

# --- Clase Principal ---
class SpriteExtractorProApp(QWidget):
    def __init__(self, config):
        super().__init__(); self.config = config; self.input_paths = []; self.current_batch_index = -1; self.is_processing_batch = False
        self.output_dir = config.get('defaults', {}).get('output_dir', "output_sprites"); self.original_extracted_sprites = []; self.processed_sprites = []
        self.extraction_worker = None; self.regeneration_worker = None; self.effects_worker = None
        self.current_working_canvas_size = config.get('defaults', {}).get('preview_size', 128); self.resampling_filter = self._get_resampling_filter(config.get('defaults', {}).get('resampling', 'LANCZOS'))
        self.initUI(); self.setAcceptDrops(True); self._update_sprite_count_label()

    def _get_resampling_filter(self, filter_name):
        m={"NEAREST": Image.Resampling.NEAREST,"BILINEAR": Image.Resampling.BILINEAR,"BICUBIC": Image.Resampling.BICUBIC,"LANCZOS": Image.Resampling.LANCZOS}; return m.get(filter_name.upper(), Image.Resampling.LANCZOS)

    def initUI(self):
        cfg_app=self.config.get('app_info',{}); cfg_ui=self.config.get('ui_text',{}); cfg_def=self.config.get('defaults',{}); cfg_opt=self.config.get('options',{}); cfg_sr=cfg_opt.get('spinbox_ranges',{}); cfg_slr=cfg_opt.get('slider_ranges',{})
        self.setWindowTitle(cfg_app.get('window_title','Sprite Extractor')); self.setGeometry(100,100,950,870)
        mh_layout=QHBoxLayout(self); mh_layout.setSpacing(15)
        left_v_layout=QVBoxLayout(); left_v_layout.setSpacing(15)
        files_gb=QGroupBox(cfg_ui.get('files_group',"1. Archivos")); files_l=QVBoxLayout(); in_l=QHBoxLayout(); self.input_display_lineedit=QLineEdit(); self.input_display_lineedit.setPlaceholderText(cfg_ui.get('input_placeholder',"Selecciona...")); self.input_display_lineedit.setReadOnly(True); self.input_button=QPushButton(cfg_ui.get('input_button',"Entrada...")); self.input_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)); self.input_button.clicked.connect(self.select_input_file); in_l.addWidget(self.input_display_lineedit,1); in_l.addWidget(self.input_button); files_l.addLayout(in_l); out_l=QHBoxLayout(); self.output_dir_lineedit=QLineEdit(self.output_dir); self.output_dir_lineedit.setPlaceholderText(cfg_ui.get('output_placeholder',"Salida...")); self.output_dir_button=QPushButton(cfg_ui.get('output_button',"Salida...")); self.output_dir_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)); self.output_dir_button.clicked.connect(self.select_output_dir); out_l.addWidget(self.output_dir_lineedit,1); out_l.addWidget(self.output_dir_button); files_l.addLayout(out_l); files_gb.setLayout(files_l); left_v_layout.addWidget(files_gb)
        extract_gb=QGroupBox(cfg_ui.get('extract_group',"2. Opciones")); extract_grid=QGridLayout(); engine_lbl=QLabel(cfg_ui.get('engine_label',"Motor:")); self.engine_combobox=QComboBox(); self.engine_combobox.addItems(cfg_opt.get('engines',["Rembg (Default)","Fine-Tuned (Placeholder)"])); self.engine_combobox.setCurrentText(cfg_def.get('engine',"Rembg (Default)")); extract_grid.addWidget(engine_lbl,0,0); extract_grid.addWidget(self.engine_combobox,0,1,1,2); canvas_lbl=QLabel(cfg_ui.get('preview_size_label',"Canvas:")); canvas_lbl.setToolTip("Tamaño lienzo trabajo"); extract_grid.addWidget(canvas_lbl,1,0,1,3); self.size_radiobuttons={}; self.size_buttongroup=QButtonGroup(self); std_sizes=cfg_opt.get('preview_sizes',[128,256,512,1024]); def_canvas=cfg_def.get('preview_size',128); rr,rc=2,0
        for s in std_sizes: rb=QRadioButton(f"{s}x{s}"); self.size_radiobuttons[s]=rb; self.size_buttongroup.addButton(rb,s); extract_grid.addWidget(rb, rr, rc); rb.setChecked(s==def_canvas); rc+=1; rr+=rc//2; rc%=2
        if not self.size_buttongroup.checkedButton() and self.size_radiobuttons: list(self.size_radiobuttons.values())[0].setChecked(True)
        cr=rr+(rc>0); self.custom_size_checkbox=QCheckBox(cfg_ui.get('custom_checkbox',"Personalizado")); self.custom_size_spinbox=QSpinBox(); cs_range=cfg_sr.get('custom_size',[16,4096]); self.custom_size_spinbox.setRange(cs_range[0],cs_range[1]); self.custom_size_spinbox.setValue(cfg_def.get('preview_size',128)); self.custom_size_spinbox.setEnabled(False); self.custom_size_checkbox.toggled.connect(self.toggle_custom_size); extract_grid.addWidget(self.custom_size_checkbox, cr, 0); extract_grid.addWidget(self.custom_size_spinbox, cr, 1, 1, 2); cr+=1
        min_area_lbl=QLabel(cfg_ui.get('min_area_label',"Área Mín:")); self.min_area_spinbox=QSpinBox(); ma_range=cfg_sr.get('min_area',[10,10000]); self.min_area_spinbox.setRange(ma_range[0],ma_range[1]); self.min_area_spinbox.setValue(cfg_def.get('min_area',100)); extract_grid.addWidget(min_area_lbl, cr, 0); extract_grid.addWidget(self.min_area_spinbox, cr, 1, 1, 2); cr+=1
        resample_lbl=QLabel(cfg_ui.get('resampling_label',"Remuestreo:")); self.resampling_combobox=QComboBox(); self.resampling_combobox.addItems(cfg_opt.get('resampling_methods',["LANCZOS","NEAREST","BILINEAR","BICUBIC"])); self.resampling_combobox.setCurrentText(cfg_def.get('resampling',"LANCZOS")); self.resampling_combobox.currentTextChanged.connect(self.update_resampling_filter); extract_grid.addWidget(resample_lbl, cr, 0); extract_grid.addWidget(self.resampling_combobox, cr, 1, 1, 2); cr+=1
        self.extract_button=QPushButton(cfg_ui.get('extract_button',"Extraer")); self.extract_button.setStyleSheet("font-weight:bold;padding:8px;"); self.extract_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)); self.extract_button.clicked.connect(self.start_extraction); extract_grid.addWidget(self.extract_button, cr, 0, 1, 3); extract_gb.setLayout(extract_grid); left_v_layout.addWidget(extract_gb)
        effects_gb=QGroupBox(cfg_ui.get('effects_group',"3. Efectos")); effects_grid=QGridLayout(); self.pixelate_checkbox=QCheckBox(cfg_ui.get('pixelate_checkbox',"Pixelizar")); self.pixelate_size_spinbox=QSpinBox(); px_range=cfg_sr.get('pixelate_size',[2,32]); self.pixelate_size_spinbox.setRange(px_range[0],px_range[1]); self.pixelate_size_spinbox.setValue(cfg_def.get('pixelate_size',4)); self.pixelate_size_spinbox.setEnabled(False); self.pixelate_checkbox.toggled.connect(self.pixelate_size_spinbox.setEnabled); effects_grid.addWidget(self.pixelate_checkbox,0,0); effects_grid.addWidget(QLabel(cfg_ui.get('pixelate_size_label',"Tamaño:")),0,1); effects_grid.addWidget(self.pixelate_size_spinbox,0,2)
        self.quantize_checkbox=QCheckBox(cfg_ui.get('quantize_checkbox',"Reducir Col")); self.quantize_colors_spinbox=QSpinBox(); qz_range=cfg_sr.get('quantize_colors',[2,256]); self.quantize_colors_spinbox.setRange(qz_range[0],qz_range[1]); self.quantize_colors_spinbox.setValue(cfg_def.get('quantize_colors',64)); self.quantize_colors_spinbox.setEnabled(False); self.quantize_method_combobox=QComboBox(); self.quantize_method_combobox.addItems(cfg_opt.get('quantize_methods',['MEDIANCUT','MAXCOVERAGE','FASTOCTREE'])); self.quantize_method_combobox.setCurrentText(cfg_def.get('quantize_method',"MEDIANCUT")); self.quantize_method_combobox.setEnabled(False); self.quantize_checkbox.toggled.connect(self.quantize_colors_spinbox.setEnabled); self.quantize_checkbox.toggled.connect(self.quantize_method_combobox.setEnabled); effects_grid.addWidget(self.quantize_checkbox,1,0); effects_grid.addWidget(QLabel(cfg_ui.get('quantize_colors_label',"Cols:")),1,1); effects_grid.addWidget(self.quantize_colors_spinbox,1,2); effects_grid.addWidget(QLabel(cfg_ui.get('quantize_method_label',"Método:")),2,1); effects_grid.addWidget(self.quantize_method_combobox,2,2)
        self.apply_effects_button=QPushButton(cfg_ui.get('apply_effects_button',"Aplicar")); self.apply_effects_button.setObjectName("ApplyEffectsButton"); self.apply_effects_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)); self.apply_effects_button.clicked.connect(self.start_apply_effects); self.apply_effects_button.setEnabled(False); self.revert_effects_button=QPushButton(cfg_ui.get('revert_effects_button',"Revertir")); self.revert_effects_button.setObjectName("RevertEffectsButton"); self.revert_effects_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)); self.revert_effects_button.clicked.connect(self.revert_to_originals); self.revert_effects_button.setEnabled(False); effects_grid.addWidget(self.apply_effects_button,3,0,1,3); effects_grid.addWidget(self.revert_effects_button,4,0,1,3); effects_gb.setLayout(effects_grid); left_v_layout.addWidget(effects_gb)
        left_v_layout.addStretch(1); mh_layout.addLayout(left_v_layout,3)
        preview_v_layout = QVBoxLayout(); preview_v_layout.setSpacing(10); preview_gb = QGroupBox(cfg_ui.get('preview_group',"4. Vista Previa")); preview_l_layout = QVBoxLayout(); self.preview_list_widget = ReorderableListWidget(); self.preview_list_widget.setObjectName("previewListWidget"); def_thumb_sz = cfg_def.get('thumbnail_size',96); self.preview_list_widget.setIconSize(QSize(def_thumb_sz,def_thumb_sz)); gw=def_thumb_sz+self.preview_list_widget.spacing()+10; gh=def_thumb_sz+self.preview_list_widget.spacing()+35; self.preview_list_widget.setGridSize(QSize(gw,gh)); self.preview_list_widget.setGridVisible(True); self.preview_list_widget.setViewMode(QListWidget.ViewMode.IconMode); self.preview_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust); self.preview_list_widget.setMovement(QListWidget.Movement.Snap); self.preview_list_widget.setSpacing(12); self.preview_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection); self.preview_list_widget.items_reordered.connect(self.sync_internal_lists_after_reorder); self.preview_list_widget.itemSelectionChanged.connect(self.update_delete_button_state); self.preview_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu); self.preview_list_widget.customContextMenuRequested.connect(self.show_preview_context_menu); preview_l_layout.addWidget(self.preview_list_widget,1)
        controls_under_list = QGridLayout(); thumb_lbl = QLabel(cfg_ui.get('thumbnail_size_label',"Zoom Thumbs:")); self.thumbnail_size_slider = QSlider(Qt.Orientation.Horizontal); thumb_sl_range = cfg_slr.get('thumbnail_size',[48,256,32]); self.thumbnail_size_slider.setRange(thumb_sl_range[0],thumb_sl_range[1]); self.thumbnail_size_slider.setValue(def_thumb_sz); self.thumbnail_size_slider.setTickInterval(thumb_sl_range[2]); self.thumbnail_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow); self.thumbnail_size_slider.valueChanged.connect(self.update_thumbnail_size_display); controls_under_list.addWidget(thumb_lbl, 0, 0); controls_under_list.addWidget(self.thumbnail_size_slider, 0, 1, 1, 2)
        self.sprite_count_label = QLabel(); self.sprite_count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter); controls_under_list.addWidget(self.sprite_count_label, 1, 0, 1, 3) # Fila 1, Span 3
        preview_btns = QHBoxLayout(); self.load_sprites_button = QPushButton(cfg_ui.get('load_button',"Cargar...")); self.load_sprites_button.setObjectName("LoadButton"); self.load_sprites_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)); self.load_sprites_button.clicked.connect(self.load_individual_sprites); preview_btns.addWidget(self.load_sprites_button); self.save_selected_button = QPushButton(cfg_ui.get('save_selected_button',"Guardar Sel...")); self.save_selected_button.setObjectName("SaveSelectedButton"); self.save_selected_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)); self.save_selected_button.setEnabled(False); self.save_selected_button.clicked.connect(self.save_selected_sprites); preview_btns.addWidget(self.save_selected_button); preview_btns.addStretch(); self.delete_button = QPushButton(cfg_ui.get('delete_button',"Eliminar Sel.")); self.delete_button.setObjectName("DeleteButton"); self.delete_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)); self.delete_button.setEnabled(False); self.delete_button.clicked.connect(self.delete_selected_sprites); preview_btns.addWidget(self.delete_button); controls_under_list.addLayout(preview_btns, 2, 0, 1, 3); preview_l_layout.addLayout(controls_under_list); preview_gb.setLayout(preview_l_layout); preview_v_layout.addWidget(preview_gb,5)
        self.output_options_groupbox = QGroupBox(cfg_ui.get('output_options_group',"5. Opciones Salida")); out_opt_l = QVBoxLayout(); ss_sub_gb = QGroupBox("Spritesheet"); ss_sub_l = QVBoxLayout(); layout_mode_l = QHBoxLayout(); layout_mode_lbl = QLabel(cfg_ui.get('regen_layout_label', "Priorizar:")); self.regen_layout_cols_radio = QRadioButton(cfg_ui.get('regen_layout_cols', "Columnas")); self.regen_layout_rows_radio = QRadioButton(cfg_ui.get('regen_layout_rows', "Filas")); self.regen_layout_group = QButtonGroup(self); self.regen_layout_group.addButton(self.regen_layout_cols_radio, 0); self.regen_layout_group.addButton(self.regen_layout_rows_radio, 1); def_layout = cfg_def.get('regen_layout_mode', "Columns");
        if def_layout == "Rows": self.regen_layout_rows_radio.setChecked(True)
        else: self.regen_layout_cols_radio.setChecked(True)
        self.regen_layout_cols_radio.toggled.connect(self._toggle_regen_spinboxes); layout_mode_l.addWidget(layout_mode_lbl); layout_mode_l.addWidget(self.regen_layout_cols_radio); layout_mode_l.addWidget(self.regen_layout_rows_radio); layout_mode_l.addStretch(); ss_sub_l.addLayout(layout_mode_l)
        ss_grid = QGridLayout(); cols_range=cfg_sr.get('regen_columns',[1,100]); self.spritesheet_columns_spinbox=QSpinBox(); self.spritesheet_columns_spinbox.setRange(cols_range[0],cols_range[1]); self.spritesheet_columns_spinbox.setValue(cfg_def.get('regen_columns',8)); ss_grid.addWidget(QLabel(cfg_ui.get('regen_columns_label',"Cols:")),0,0); ss_grid.addWidget(self.spritesheet_columns_spinbox,0,1)
        rows_range=cfg_sr.get('regen_rows',[1,100]); self.spritesheet_rows_spinbox=QSpinBox(); self.spritesheet_rows_spinbox.setRange(rows_range[0],rows_range[1]); self.spritesheet_rows_spinbox.setValue(cfg_def.get('regen_rows',8)); ss_grid.addWidget(QLabel(cfg_ui.get('regen_rows_label',"Filas:")),1,0); ss_grid.addWidget(self.spritesheet_rows_spinbox,1,1)
        self.spritesheet_spacing_spinbox=QSpinBox(); sp_range=cfg_sr.get('regen_spacing',[0,100]); self.spritesheet_spacing_spinbox.setRange(sp_range[0],sp_range[1]); self.spritesheet_spacing_spinbox.setValue(cfg_def.get('regen_spacing',2)); ss_grid.addWidget(QLabel(cfg_ui.get('regen_spacing_label',"Esp:")),2,0); ss_grid.addWidget(self.spritesheet_spacing_spinbox,2,1); ss_sub_l.addLayout(ss_grid); self._toggle_regen_spinboxes(self.regen_layout_cols_radio.isChecked()); self.generate_spritesheet_button=QPushButton(cfg_ui.get('regen_button',"Crear Sheet")); self.generate_spritesheet_button.setStyleSheet("font-weight:bold;padding:8px;"); self.generate_spritesheet_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon)); self.generate_spritesheet_button.clicked.connect(self.start_regeneration); ss_sub_l.addWidget(self.generate_spritesheet_button); ss_sub_gb.setLayout(ss_sub_l); out_opt_l.addWidget(ss_sub_gb)
        resize_gb = QGroupBox(cfg_ui.get('resize_output_group',"Redim. Salida")); resize_l = QVBoxLayout(); self.resize_output_checkbox = QCheckBox(cfg_ui.get('resize_output_checkbox', "Redimensionar")); self.resize_output_checkbox.setChecked(cfg_def.get('resize_output_enabled', False)); resize_l.addWidget(self.resize_output_checkbox); resize_grid = QGridLayout()
        self.resize_output_width_spinbox=QSpinBox(); rsz_w_range = cfg_sr.get('resize_output_width', [8, 4096]); self.resize_output_width_spinbox.setRange(rsz_w_range[0], rsz_w_range[1]); self.resize_output_width_spinbox.setValue(cfg_def.get('resize_output_width', 64)); resize_grid.addWidget(QLabel(cfg_ui.get('resize_output_width_label',"Ancho:")), 0, 0); resize_grid.addWidget(self.resize_output_width_spinbox, 0, 1)
        self.resize_output_height_spinbox=QSpinBox(); rsz_h_range = cfg_sr.get('resize_output_height', [8, 4096]); self.resize_output_height_spinbox.setRange(rsz_h_range[0], rsz_h_range[1]); self.resize_output_height_spinbox.setValue(cfg_def.get('resize_output_height', 64)); resize_grid.addWidget(QLabel(cfg_ui.get('resize_output_height_label',"Alto:")), 1, 0); resize_grid.addWidget(self.resize_output_height_spinbox, 1, 1)
        self.resize_output_resampling_combobox=QComboBox(); self.resize_output_resampling_combobox.addItems(cfg_opt.get('resampling_methods',["LANCZOS","NEAREST"])); self.resize_output_resampling_combobox.setCurrentText(cfg_def.get('resize_output_resampling',"LANCZOS")); resize_grid.addWidget(QLabel(cfg_ui.get('resize_output_resampling_label',"Remuestreo:")), 2, 0); resize_grid.addWidget(self.resize_output_resampling_combobox, 2, 1); resize_l.addLayout(resize_grid)
        self.resize_output_width_spinbox.setEnabled(self.resize_output_checkbox.isChecked()); self.resize_output_height_spinbox.setEnabled(self.resize_output_checkbox.isChecked()); self.resize_output_resampling_combobox.setEnabled(self.resize_output_checkbox.isChecked()); self.resize_output_checkbox.toggled.connect(self.resize_output_width_spinbox.setEnabled); self.resize_output_checkbox.toggled.connect(self.resize_output_height_spinbox.setEnabled); self.resize_output_checkbox.toggled.connect(self.resize_output_resampling_combobox.setEnabled)
        resize_gb.setLayout(resize_l); out_opt_l.addWidget(resize_gb); self.output_options_groupbox.setLayout(out_opt_l); preview_v_layout.addWidget(self.output_options_groupbox, 2); self.output_options_groupbox.setEnabled(False)
        mh_layout.addLayout(preview_v_layout,5)
        log_logo_v_layout = QVBoxLayout(); log_gb = QGroupBox(cfg_ui.get('log_group',"Registro")); log_l = QVBoxLayout(); self.log_output_textedit = QTextEdit(); self.log_output_textedit.setReadOnly(True); self.log_output_textedit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth); log_l.addWidget(self.log_output_textedit); log_gb.setLayout(log_l); log_logo_v_layout.addWidget(log_gb,1); self.progress_bar = QProgressBar(); self.progress_bar.setRange(0,100); self.progress_bar.setValue(0); self.progress_bar.setTextVisible(True); self.progress_bar.setVisible(False); log_logo_v_layout.addWidget(self.progress_bar,0); log_logo_v_layout.addStretch(1); self.logo_label = QLabel(); self.logo_label.setObjectName("LogoLabel")
        logo_path = cfg_app.get('logo_filename',"app_logo.png"); logo_fpath = logo_path
        if not os.path.isabs(logo_path):
            try: script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError: script_dir = os.getcwd()
            logo_fpath = os.path.join(script_dir, logo_path)
        logo_pix = QPixmap(logo_fpath);
        if logo_pix.isNull(): self.update_log_output(f"Warn: Logo no cargado '{logo_fpath}'."); logo_pix = self.create_placeholder_logo(120,40)
        else: max_w = cfg_app.get('max_logo_width',180);
        if logo_pix.width() > max_w: logo_pix = logo_pix.scaledToWidth(max_w, Qt.TransformationMode.SmoothTransformation)
        self.logo_label.setPixmap(logo_pix); self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter); log_logo_v_layout.addWidget(self.logo_label); mh_layout.addLayout(log_logo_v_layout,3)
        self.setLayout(mh_layout); self.apply_styles(); self.set_buttons_enabled_status(True)

    # --- Métodos restantes (incluyen correcciones de sintaxis y lógica) ---
    def _toggle_regen_spinboxes(self, prioritize_cols_checked):
        parent_is_enabled = self.output_options_groupbox.isEnabled()
        self.spritesheet_columns_spinbox.setEnabled(prioritize_cols_checked and parent_is_enabled)
        self.spritesheet_rows_spinbox.setEnabled(not prioritize_cols_checked and parent_is_enabled)

    def create_placeholder_logo(self, width, height):
        pixmap = QPixmap(width,height); pixmap.fill(Qt.GlobalColor.transparent); painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing); painter.setBrush(QColor("#444")); painter.setPen(Qt.PenStyle.NoPen); painter.drawRect(pixmap.rect()); painter.setPen(QColor("#ccc")); font = painter.font(); font.setPointSize(10); font.setBold(True); painter.setFont(font); painter.drawText(pixmap.rect(),Qt.AlignmentFlag.AlignCenter,"Logo Error"); painter.end(); return pixmap

    def apply_styles(self):
        script_directory = "";
        try: script_directory = os.path.dirname(os.path.abspath(__file__))
        except NameError: script_directory = os.getcwd()
        default_stylesheet_rel_path = "styles/dark_theme.qss"; stylesheet_relative_path = self.config.get("app_info", {}).get("stylesheet_path", default_stylesheet_rel_path)
        style_file_path = stylesheet_relative_path if os.path.isabs(stylesheet_relative_path) else os.path.join(script_directory, stylesheet_relative_path)
        qss_content = load_stylesheet_from_file(style_file_path)
        if qss_content: self.setStyleSheet(qss_content)
        else: self.update_log_output("ADVERTENCIA: No se pudo cargar QSS. Usando fallback.")

    def _update_sprite_count_label(self):
        count = self.preview_list_widget.count()
        label_format = self.config.get("ui_text", {}).get("sprite_count_label", "Sprites: {count}")
        self.sprite_count_label.setText(label_format.format(count=count))

    @pyqtSlot(bool)
    def toggle_custom_size(self, checked):
        self.custom_size_spinbox.setEnabled(checked)
        for radio_btn in self.size_radiobuttons.values(): radio_btn.setEnabled(not checked)
        default_preview_size = self.config.get('defaults',{}).get('preview_size',128)
        if not checked and self.size_buttongroup.checkedButton() is None:
            if default_preview_size in self.size_radiobuttons: self.size_radiobuttons[default_preview_size].setChecked(True)
            elif self.size_radiobuttons: list(self.size_radiobuttons.values())[0].setChecked(True)

    @pyqtSlot()
    def select_input_file(self):
        if self.is_currently_busy(): return
        dialog_result = QFileDialog.getOpenFileNames(self,'Seleccionar Imagen o Lote','','Imágenes (*.png *.jpg *.jpeg *.bmp *.webp);;Todos (*)')
        filenames = dialog_result[0]
        if filenames: self.process_selected_input_files(filenames)

    @pyqtSlot()
    def select_output_dir(self):
        current_dir = self.output_dir_lineedit.text() if self.output_dir_lineedit.text() else self.output_dir
        directory_name = QFileDialog.getExistingDirectory(self, 'Directorio de Salida', current_dir)
        if directory_name: self.output_dir = directory_name; self.output_dir_lineedit.setText(directory_name); self.update_log_output(f"Salida establecida: {directory_name}")

    @pyqtSlot(str)
    def update_log_output(self, message):
        self.log_output_textedit.append(message)
        try: self.log_output_textedit.verticalScrollBar().setValue(self.log_output_textedit.verticalScrollBar().maximum())
        except Exception: pass

    @pyqtSlot(int)
    def update_progress_bar_value(self, value):
        if self.progress_bar.isVisible(): self.progress_bar.setValue(max(0,min(100,value)))

    def show_message_box(self, title, message, level="info"):
        if level == "error": QMessageBox.critical(self, title, message)
        elif level == "warning": QMessageBox.warning(self, title, message)
        else: QMessageBox.information(self, title, message)

    def clear_preview_and_sprite_data(self):
        self.preview_list_widget.clear(); self.original_extracted_sprites.clear(); self.processed_sprites.clear()
        self.update_log_output("Vista previa y datos limpiados."); self._update_sprite_count_label()
        self.set_buttons_enabled_status(not self.is_currently_busy())
        if self.progress_bar.isVisible(): self.progress_bar.setValue(0); self.progress_bar.setVisible(False)

    def process_selected_input_files(self, filenames):
        valid_extensions = ('.png','.jpg','.jpeg','.bmp','.webp'); valid_paths = []; invalid_files_info = []
        for filename in filenames:
            if isinstance(filename, str) and os.path.isfile(filename) and filename.lower().endswith(valid_extensions): valid_paths.append(filename)
            else: invalid_files_info.append(os.path.basename(filename) if isinstance(filename, str) else "?")
        if invalid_files_info: self.update_log_output(f"Adv: Ignorados: {', '.join(invalid_files_info)}"); self.show_message_box("Aviso", f"Ignorados {len(invalid_files_info)} archivo(s).","warning")
        if not valid_paths: self.show_message_box("Error","No archivos válidos.","error"); self.input_paths = []; self.input_display_lineedit.setText(""); return
        self.input_paths = valid_paths; num_files = len(valid_paths); display_text = os.path.basename(valid_paths[0]) if num_files == 1 else f"{num_files} archivos"
        self.input_display_lineedit.setText(display_text); self.update_log_output(f"Entrada: {display_text}")

    def is_currently_busy(self):
        return ((self.extraction_worker and self.extraction_worker.isRunning()) or (self.regeneration_worker and self.regeneration_worker.isRunning()) or (self.effects_worker and self.effects_worker.isRunning()) or (self.is_processing_batch and self.current_batch_index >= 0))

    def get_selected_canvas_size_value(self):
        if self.custom_size_checkbox.isChecked(): return self.custom_size_spinbox.value()
        checked_id = self.size_buttongroup.checkedId(); return checked_id if checked_id > 0 else 128

    @pyqtSlot(str)
    def update_resampling_filter(self, filter_name_text): self.resampling_filter = self._get_resampling_filter(filter_name_text)

    @pyqtSlot()
    def start_extraction(self):
        self.update_log_output("--- Extracción ---");
        if self.is_currently_busy(): self.show_message_box("Aviso","Ocupado.","warning"); return
        if not self.input_paths: self.show_message_box("Error","Selecciona imagen.","error"); return
        try:
            self.current_working_canvas_size = self.get_selected_canvas_size_value(); self.update_resampling_filter(self.resampling_combobox.currentText())
            if self.current_working_canvas_size < 16: self.show_message_box("Error","Canvas >= 16px.","error"); return
            self.clear_preview_and_sprite_data(); n=len(self.input_paths); self.update_log_output(f"Iniciando {n} archivo(s)..."); self.update_log_output(f"Canvas: {self.current_working_canvas_size}px | Filtro: {self.resampling_combobox.currentText()}")
            self.set_buttons_enabled_status(False); self.progress_bar.setRange(0,100); self.progress_bar.setValue(0); self.progress_bar.setVisible(True); self.is_processing_batch=True; self.current_batch_index=0
            self._start_single_file_extraction(self.current_batch_index)
        except Exception as e: self.update_log_output(f"Error inicio extr: {e}\n{traceback.format_exc()}"); self.show_message_box("Error",f"Error: {e}","error"); self.is_processing_batch=False; self.current_batch_index=-1; self.set_buttons_enabled_status(True); self.progress_bar.setVisible(False)

    def _start_single_file_extraction(self, file_index_in_batch):
        if not self.is_processing_batch or not (0 <= file_index_in_batch < len(self.input_paths)):
            if self.is_processing_batch: self._handle_batch_extraction_finished(); return
        current_file_path = self.input_paths[file_index_in_batch]
        self.update_log_output(f"--- Procesando archivo {file_index_in_batch + 1}/{len(self.input_paths)}: {os.path.basename(current_file_path)} ---")
        if self.extraction_worker: self.extraction_worker.quit(); self.extraction_worker.wait(100); self.extraction_worker.deleteLater(); self.extraction_worker = None
        selected_engine_instance = None
        try:
            engine_name_text = self.engine_combobox.currentText()
            if "Rembg" in engine_name_text: selected_engine_instance = RembgEngine()
            elif "Fine-Tuned" in engine_name_text: fine_tuned_model_path = self.config.get('models',{}).get('fine_tuned_path',"?"); selected_engine_instance = PlaceholderFineTunedEngine(model_path=fine_tuned_model_path)
            else: raise ValueError(f"Motor desconocido: {engine_name_text}")
            self.update_log_output(f"  Motor: {engine_name_text.split(' (')[0]}")
        except Exception as e: self._handle_single_extraction_error(f"Error inicializando motor: {e}"); return
        try:
            min_contour_area = self.min_area_spinbox.value(); thumb_qsize = self.preview_list_widget.iconSize(); thumbnail_pil_size_tuple = (thumb_qsize.width(), thumb_qsize.height())
            if not (thumbnail_pil_size_tuple[0] > 0 and thumbnail_pil_size_tuple[1] > 0): thumbnail_pil_size_tuple = (96, 96); self.update_log_output("Warn: Thumb size inválido, usando 96x96.")
            self.update_log_output(f"  Área mín={min_contour_area}px²")
            self.extraction_worker = ExtractionWorker(selected_engine_instance, current_file_path, min_contour_area, self.current_working_canvas_size, self.resampling_filter, thumbnail_pil_size_tuple)
            self.extraction_worker.progress.connect(self.update_log_output, Qt.ConnectionType.QueuedConnection); self.extraction_worker.thumbnail_ready.connect(self.add_thumbnail_to_preview_list, Qt.ConnectionType.QueuedConnection); self.extraction_worker.extraction_data_ready.connect(self.receive_extracted_sprite_data, Qt.ConnectionType.QueuedConnection); self.extraction_worker.error.connect(self._handle_single_extraction_error, Qt.ConnectionType.QueuedConnection); self.extraction_worker.finished.connect(self._handle_single_extraction_finished_with_message, Qt.ConnectionType.QueuedConnection); self.extraction_worker.progress_update.connect(self.update_progress_bar_value, Qt.ConnectionType.QueuedConnection); self.extraction_worker.finished.connect(self._cleanup_extraction_qthread_object)
            self.update_log_output("  Iniciando worker..."); self.progress_bar.setValue(0); self.extraction_worker.start()
        except Exception as e: self._handle_single_extraction_error(f"Error preparando worker: {e}"); self._clear_and_nullify_extraction_worker()

    @pyqtSlot()
    def _cleanup_extraction_qthread_object(self):
        sender_worker = self.sender();
        if sender_worker and isinstance(sender_worker, ExtractionWorker): sender_worker.deleteLater();
        if self.extraction_worker == sender_worker: self.extraction_worker = None

    @pyqtSlot(str)
    def _handle_single_extraction_error(self, error_message):
        if not self.is_processing_batch or self.current_batch_index < 0: return
        current_filename = f"Archivo {self.current_batch_index + 1}"
        if 0 <= self.current_batch_index < len(self.input_paths): current_filename = os.path.basename(self.input_paths[self.current_batch_index])
        full_error_message = f"ERROR ({current_filename}):\n{error_message}"; self.update_log_output(full_error_message); self.show_message_box(f"Error ({current_filename})", error_message, "error")
        if self.extraction_worker: self.extraction_worker.deleteLater(); self._clear_and_nullify_extraction_worker()
        QTimer.singleShot(0, lambda: self._handle_single_extraction_finished_with_message(f"Finalizado con error: {current_filename}"))

    @pyqtSlot(str)
    def _handle_single_extraction_finished_with_message(self, message_from_worker):
        if not self.is_processing_batch: return
        if self.extraction_worker: self.extraction_worker = None
        if self.current_batch_index < 0: self._handle_batch_extraction_finished(); return
        self.update_log_output(f"  Resultado archivo {self.current_batch_index + 1}: {message_from_worker}"); self.update_log_output(f"--- Archivo {self.current_batch_index + 1} procesado ---")
        next_file_index = self.current_batch_index + 1
        if next_file_index < len(self.input_paths):
            self.current_batch_index = next_file_index; self.update_log_output(f"Siguiente ({self.current_batch_index + 1}/{len(self.input_paths)})..."); QTimer.singleShot(100, lambda: self._start_single_file_extraction(self.current_batch_index))
        else:
            self.current_batch_index = -1; QTimer.singleShot(100, self._handle_batch_extraction_finished)

    def _handle_batch_extraction_finished(self):
        if not self.is_processing_batch: return
        self.update_log_output("--- LOTE OK ---"); self.is_processing_batch = False; self.current_batch_index = -1; self._clear_and_nullify_extraction_worker()
        has_sprites = bool(self.processed_sprites); final_msg = f"Lote OK. {self.preview_list_widget.count()} sprites." if has_sprites else "Lote OK. No sprites."
        self.show_message_box("Lote Completado", final_msg, "info" if has_sprites else "warning"); self.progress_bar.setVisible(False); self.set_buttons_enabled_status(True); self.update_log_output("Controles ON.")

    @pyqtSlot()
    def _clear_and_nullify_extraction_worker(self): self.extraction_worker = None
    @pyqtSlot()
    def _clear_and_nullify_regeneration_worker(self): self.regeneration_worker = None
    @pyqtSlot()
    def _clear_and_nullify_effects_worker(self): self.effects_worker = None

    @pyqtSlot(object)
    def add_thumbnail_to_preview_list(self, thumbnail_image_pil):
        if not isinstance(thumbnail_image_pil, Image.Image): self.update_log_output("Error: Add thumb no válido."); return
        try:
            visual_index = self.preview_list_widget.count(); data_index = visual_index; canvas_size = self.current_working_canvas_size
            final_thumb = thumbnail_image_pil
            if final_thumb.mode!='RGBA': temp = final_thumb.copy(); final_thumb = temp.convert('RGB').convert('RGBA') if temp.mode == 'P' else temp.convert('RGBA')
            qimg = ImageQt.ImageQt(final_thumb); pixmap = QPixmap.fromImage(qimg)
            if pixmap.isNull(): pixmap = QPixmap(self.preview_list_widget.iconSize()); pixmap.fill(Qt.GlobalColor.red); self.update_log_output(f"Warn: Falló QPixmap thumb {visual_index + 1}.")
            icon = QIcon(pixmap); item_text = f"Sprite {visual_index + 1}\n({canvas_size}x{canvas_size})"; list_item = QListWidgetItem(icon, item_text)
            icon_h, icon_w = self.preview_list_widget.iconSize().height(), self.preview_list_widget.iconSize().width(); list_item.setSizeHint(QSize(max(icon_w + 10, 80), icon_h + 40)); list_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom); list_item.setData(Qt.ItemDataRole.UserRole, data_index); list_item.setToolTip(f"Índice: {data_index}\nCanvas: {canvas_size}x{canvas_size}")
            self.preview_list_widget.addItem(list_item); self._update_sprite_count_label()
        except Exception as e: idx_str = str(data_index) if 'data_index' in locals() else '?'; self.update_log_output(f"Err add thumb {idx_str}: {e}\n{traceback.format_exc()}")

    @pyqtSlot(list)
    def receive_extracted_sprite_data(self, extracted_canvases_list):
        n=len(extracted_canvases_list); self.update_log_output(f"Recibidos {n} canvases.");
        self.original_extracted_sprites.extend([img.copy() for img in extracted_canvases_list if isinstance(img,Image.Image)])
        self.processed_sprites.extend([img.copy() for img in extracted_canvases_list if isinstance(img,Image.Image)])
        self.update_log_output(f"Datos actualizados. Total: {len(self.processed_sprites)}")
        self.set_buttons_enabled_status(not self.is_currently_busy())

    @pyqtSlot()
    def update_delete_button_state(self):
        has_selection = len(self.preview_list_widget.selectedItems()) > 0; can_interact = has_selection and not self.is_currently_busy()
        self.delete_button.setEnabled(can_interact); self.save_selected_button.setEnabled(can_interact)

    @pyqtSlot()
    def delete_selected_sprites(self):
        self.update_log_output("--- Del Sel ---");
        if self.is_currently_busy(): self.show_message_box("Aviso", "Ocupado.", "warning"); return
        selected_items = self.preview_list_widget.selectedItems(); n = len(selected_items)
        if not selected_items: return;
        self.update_log_output(f"  {n} selecc.")
        cfg_ui = self.config.get('ui_text', {}); reply = QMessageBox.question(self, cfg_ui.get('confirm_delete_title','Confirmar'), cfg_ui.get('confirm_delete_text','Del {num}?').format(num=n), QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No: return
        self.set_buttons_enabled_status(False); indices_to_delete = set(); op_ok = False
        for item in selected_items: idx = item.data(Qt.ItemDataRole.UserRole);
        if idx is not None and isinstance(idx, int): indices_to_delete.add(idx)
        if not indices_to_delete: self.update_log_output("Err: No índices válidos."); op_ok=True;
        else:
            sorted_data_indices = sorted(list(indices_to_delete), reverse=True); deleted = 0
            try:
                if len(self.original_extracted_sprites) != len(self.processed_sprites): raise RuntimeError("¡Desinc listas!")
                for idx in sorted_data_indices:
                    if 0 <= idx < len(self.original_extracted_sprites): del self.original_extracted_sprites[idx]; del self.processed_sprites[idx]; deleted += 1
                self.preview_list_widget.clear(); self.update_log_output("Actualizando vista post-delete...")
                for new_idx, sprite_canvas in enumerate(self.processed_sprites):
                    try:
                        thumb_qsize = self.preview_list_widget.iconSize(); ts = (thumb_qsize.width(), thumb_qsize.height()); ts = ts if ts[0]>0 and ts[1]>0 else (64,64)
                        thumb = sprite_canvas.resize(ts, Image.Resampling.LANCZOS); self.add_thumbnail_to_preview_list(thumb)
                    except Exception as e_rethumb: self.update_log_output(f"Err rethumb post-del {new_idx}: {e_rethumb}")
                self.update_log_output(f"--- Borrado OK ({deleted}) ---"); op_ok = True
            except Exception as e: self.update_log_output(f"¡ERROR delete!\n{e}\n{traceback.format_exc()}"); self.show_message_box("Error", f"Error: {e}", "error"); op_ok = False
        self._update_sprite_count_label(); self.set_buttons_enabled_status(True)
        if op_ok and len(self.processed_sprites) != self.preview_list_widget.count(): self.update_log_output("¡ERR POST-DEL! Desinc."); self.show_message_box("Error", "Inconsistencia post-borrado.", "warning")
        self.update_delete_button_state()

    @pyqtSlot()
    def load_individual_sprites(self):
        self.update_log_output("--- Carga Indiv ---");
        if self.is_currently_busy(): return
        fns_tuple = QFileDialog.getOpenFileNames(self,'Cargar Sprites','','Imágenes (*.png *.jpg *.jpeg *.bmp *.webp);;Todos (*)'); fns = fns_tuple[0]
        if not fns: return; self.update_log_output(f"--- Cargando {len(fns)}... ---")
        try: tcs = self.get_selected_canvas_size_value(); self.update_resampling_filter(self.resampling_combobox.currentText()); rf = self.resampling_filter; tq = self.preview_list_widget.iconSize(); tp = (tq.width(), tq.height()); tp = tp if tp[0]>0 and tp[1]>0 else (96,96)
        except Exception as e: self.update_log_output(f"Err config carga: {e}. Defaults."); tcs=128; rf=Image.Resampling.LANCZOS; tp=(96,96)
        loaded=0; self.set_buttons_enabled_status(False); self.progress_bar.setRange(0,len(fns)); self.progress_bar.setValue(0); self.progress_bar.setVisible(True)
        try:
            for i, fp in enumerate(fns):
                try:
                    img = Image.open(fp); img_rgba = img
                    if img.mode != 'RGBA': img_rgba = img.convert('RGB').convert('RGBA') if img.mode == 'P' else img.convert('RGBA')
                    canvas=Image.new('RGBA',(tcs,tcs),(0,0,0,0)); fw,fh=img_rgba.size;
                    if fw<=0 or fh<=0: self.update_log_output(f"  Adv: Dim inválidas {os.path.basename(fp)}. Skip."); self.progress_bar.setValue(i+1); continue
                    m=0.95; s=min((tcs*m)/fw,(tcs*m)/fh); s=max(0.001,s); nw,nh=int(fw*s),int(fh*s); nw=max(1,nw); nh=max(1,nh);
                    rz=img_rgba.resize((nw,nh),resample=rf); px,py=(tcs-nw)//2,(tcs-nh)//2;
                    if rz.mode != 'RGBA': rz = rz.convert('RGB').convert('RGBA') if rz.mode == 'P' else rz.convert('RGBA')
                    canvas.paste(rz,(px,py),rz); thumb=canvas.resize(tp,Image.Resampling.LANCZOS);
                    self.original_extracted_sprites.append(canvas.copy()); self.processed_sprites.append(canvas.copy()); self.add_thumbnail_to_preview_list(thumb); loaded+=1
                except Exception as e_load: self.update_log_output(f"  Err load/proc {os.path.basename(fp)}: {e_load}")
                finally: self.progress_bar.setValue(i + 1)
        finally:
            self.progress_bar.setVisible(False); self.set_buttons_enabled_status(True); self.update_log_output(f"--- Carga OK ({loaded}) ---"); self._update_sprite_count_label()

    # --- MÉTODO SAVE CORREGIDO ---
    @pyqtSlot()
    def save_selected_sprites(self):
        self.update_log_output("--- Iniciando guardado de sprites seleccionados ---")
        if self.is_currently_busy(): return
        selected_items = self.preview_list_widget.selectedItems(); num_selected = len(selected_items)
        if not selected_items: self.show_message_box("Información","No hay sprites seleccionados.","info"); return

        save_directory = QFileDialog.getExistingDirectory(self,'Seleccionar Directorio para Guardar Sprites', self.output_dir)
        if not save_directory: return

        resize_enabled = self.resize_output_checkbox.isChecked(); target_w=0; target_h=0; resize_f=None; errors=0
        if resize_enabled:
            target_w = self.resize_output_width_spinbox.value(); target_h = self.resize_output_height_spinbox.value()
            resize_f_name = self.resize_output_resampling_combobox.currentText(); resize_f = self._get_resampling_filter(resize_f_name)
            self.update_log_output(f"Redimensionando salida a {target_w}x{target_h} ({resize_f_name})")
            if target_w <= 0 or target_h <= 0: self.show_message_box("Error", "Dimensiones resize inválidas.", "error"); return

        base_filename_prefix = "sprite"
        if self.input_paths:
             # --- CORRECCIÓN DE SINTAXIS APLICADA ---
             try:
                 base_filename_prefix = os.path.splitext(os.path.basename(self.input_paths[0]))[0]
             except (IndexError, TypeError):
                 base_filename_prefix = "sprite_lote"
             # --- FIN CORRECCIÓN ---
        elif self.original_extracted_sprites:
             base_filename_prefix = "sprite_cargado"

        self.update_log_output(f"--- Guardando {num_selected} sprite(s) en '{save_directory}' ---")
        saved_count = 0;
        self.set_buttons_enabled_status(False); self.progress_bar.setRange(0, num_selected); self.progress_bar.setValue(0); self.progress_bar.setVisible(True)
        try:
            for i, list_item in enumerate(selected_items):
                data_index = list_item.data(Qt.ItemDataRole.UserRole)
                if data_index is not None and 0 <= data_index < len(self.processed_sprites):
                    try:
                        sprite_to_process = self.processed_sprites[data_index]
                        sprite_final = sprite_to_process
                        if resize_enabled:
                             sprite_final = sprite_to_process.resize((target_w, target_h), resample=resize_f)

                        output_filename = f"{base_filename_prefix}_sprite_{data_index + 1:03d}.png"
                        full_save_path = os.path.join(save_directory, output_filename)
                        spr_save = sprite_final
                        if spr_save.mode!='RGBA':
                            spr_save = spr_save.convert('RGB').convert('RGBA') if spr_save.mode == 'P' else spr_save.convert('RGBA')
                        spr_save.save(full_save_path,"PNG")
                        self.update_log_output(f"  Guardado: {output_filename}" + (" (redim)" if resize_enabled else ""))
                        saved_count+=1;
                    except Exception as e_save:
                        self.update_log_output(f"  Error al guardar/redimensionar sprite {data_index}: {e_save}"); errors+=1;
                else:
                    self.update_log_output(f"  Advertencia: Índice de datos inválido o sprite no encontrado: {data_index}"); errors+=1;
                self.progress_bar.setValue(i+1);
        finally:
            self.progress_bar.setVisible(False); self.set_buttons_enabled_status(True)
            self.update_log_output(f"--- Guardado ({saved_count} OK, {errors} err) ---")
            summary = f"{saved_count} OK"+(f", {errors} err." if errors else ".")
            self.show_message_box("Guardado",f"{summary}\nDir: {save_directory}","warning" if errors else "info")

    @pyqtSlot()
    def sync_internal_lists_after_reorder(self):
        self.update_log_output("--- Sync post-reorder ---");
        if self.is_currently_busy(): return;
        vis_c, orig_c, proc_c = self.preview_list_widget.count(),len(self.original_extracted_sprites),len(self.processed_sprites);
        if orig_c!=proc_c or vis_c!=orig_c: self.update_log_output(f"¡ERR CRIT! Desync: V={vis_c},O={orig_c},P={proc_c}"); self.show_message_box("Error","Desync. Reiniciar.","error"); return;
        new_orig, new_proc = [None]*vis_c, [None]*vis_c; valid = True; v2old_map = {}
        try:
            for vr in range(vis_c):
                item = self.preview_list_widget.item(vr);
                if item is None: valid = False; self.update_log_output(f"Err: Ítem nulo {vr}."); break
                old_idx = item.data(Qt.ItemDataRole.UserRole)
                if old_idx is None or not isinstance(old_idx, int) or not (0 <= old_idx < orig_c): valid = False; self.update_log_output(f"Err: Índice inválido ({old_idx}) fila {vr}."); break
                v2old_map[vr] = old_idx
            if not valid: raise ValueError("Índices inválidos.")
            for nv_idx in range(vis_c): old_idx = v2old_map[nv_idx]; new_orig[nv_idx] = self.original_extracted_sprites[old_idx]; new_proc[nv_idx] = self.processed_sprites[old_idx]
            self.original_extracted_sprites = new_orig; self.processed_sprites = new_proc; self.update_log_output("Actualizando UI indices...")
            for cv_idx in range(vis_c):
                item = self.preview_list_widget.item(cv_idx);
                if item is None: continue
                new_idx = cv_idx; item.setData(Qt.ItemDataRole.UserRole, new_idx)
                cs_str = f"{self.current_working_canvas_size}x{self.current_working_canvas_size}";
                try: parts = item.text().split('\n'); cs_str = parts[1].strip('()') if len(parts) > 1 else cs_str
                except: pass
                item.setText(f"Sprite {new_idx + 1}\n({cs_str})"); item.setToolTip(f"Índice: {new_idx}\nCanvas: {cs_str}");
            self.update_log_output("--- Sync OK ---");
        except Exception as e: self.update_log_output(f"¡ERR sync!\n{e}\n{traceback.format_exc()}"); self.show_message_box("Error",f"Err sync: {e}","error");

    @pyqtSlot(int)
    def update_thumbnail_size_display(self, value):
        if value<=0: return; nts=QSize(value,value); self.preview_list_widget.setIconSize(nts); gw=value+self.preview_list_widget.spacing()+10; gh=value+self.preview_list_widget.spacing()+35; self.preview_list_widget.setGridSize(QSize(gw,gh)); self.preview_list_widget.viewport().update();

    def _get_selected_single_sprite_index(self):
        selected = self.preview_list_widget.selectedItems();
        if len(selected) == 1:
            item = selected[0]; data_index = item.data(Qt.ItemDataRole.UserRole)
            if data_index is not None and isinstance(data_index, int) and 0 <= data_index < len(self.processed_sprites): return data_index
            else: self.update_log_output(f"Error: Índice ({data_index}) fuera rango (0-{len(self.processed_sprites)-1})."); return None
        return None

    def _apply_transformation(self, data_index, transformation_func):
        if data_index is None or self.is_currently_busy(): return
        try:
            self.update_log_output(f"Transformando sprite {data_index + 1}..."); self.set_buttons_enabled_status(False)
            transformed = transformation_func(self.processed_sprites[data_index])
            self.processed_sprites[data_index] = transformed; self.update_processed_sprite_and_preview(data_index, transformed)
            self.update_log_output(f"Transformación OK sprite {data_index + 1}.")
        except Exception as e: self.update_log_output(f"Err transform {data_index + 1}: {e}"); self.show_message_box("Error", f"Err transform: {e}", "error")
        finally: QTimer.singleShot(50, lambda: self.set_buttons_enabled_status(True))

    @pyqtSlot()
    def flip_sprite_horizontal(self): idx = self._get_selected_single_sprite_index(); self._apply_transformation(idx, lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    @pyqtSlot()
    def flip_sprite_vertical(self): idx = self._get_selected_single_sprite_index(); self._apply_transformation(idx, lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    @pyqtSlot()
    def rotate_sprite_right_90(self): idx = self._get_selected_single_sprite_index(); self._apply_transformation(idx, lambda img: img.transpose(Image.Transpose.ROTATE_270))
    @pyqtSlot()
    def rotate_sprite_left_90(self): idx = self._get_selected_single_sprite_index(); self._apply_transformation(idx, lambda img: img.transpose(Image.Transpose.ROTATE_90))
    @pyqtSlot()
    def duplicate_sprite(self):
        idx = self._get_selected_single_sprite_index();
        if idx is None or self.is_currently_busy(): return
        try:
            self.update_log_output(f"Duplicando sprite {idx + 1}..."); self.set_buttons_enabled_status(False)
            orig_copy = self.original_extracted_sprites[idx].copy(); proc_copy = self.processed_sprites[idx].copy()
            insert_pos = idx + 1; self.original_extracted_sprites.insert(insert_pos, orig_copy); self.processed_sprites.insert(insert_pos, proc_copy)
            self.update_log_output(f"Duplicado insertado en datos pos {insert_pos + 1}."); self.preview_list_widget.clear(); self.update_log_output("Actualizando vista post-duplicar...")
            for i, spr in enumerate(self.processed_sprites):
                try: tq = self.preview_list_widget.iconSize(); ts=(tq.width(),tq.height()); ts = ts if ts[0]>0 and ts[1]>0 else (64,64); thumb = spr.resize(ts, Image.Resampling.LANCZOS); self.add_thumbnail_to_preview_list(thumb)
                except Exception as e_rethumb: self.update_log_output(f"Err rethumb post-duplicar {i}: {e_rethumb}")
            self.update_log_output("Duplicado OK.");
            if insert_pos < self.preview_list_widget.count(): self.preview_list_widget.setCurrentRow(insert_pos)
        except Exception as e: self.update_log_output(f"Err duplicar {idx + 1}: {e}\n{traceback.format_exc()}"); self.show_message_box("Error", f"Err duplicar: {e}", "error")
        finally: QTimer.singleShot(50, lambda: self.set_buttons_enabled_status(True))


    @pyqtSlot(QPoint)
    def show_preview_context_menu(self, position):
        if self.is_currently_busy(): return
        selected = self.preview_list_widget.selectedItems(); n_sel = len(selected); single_sel = (n_sel == 1); menu = QMenu()
        if single_sel:
            t_menu = QMenu("Transformar", self);
            fh=QAction("Voltear H",self); fh.triggered.connect(self.flip_sprite_horizontal); t_menu.addAction(fh)
            fv=QAction("Voltear V",self); fv.triggered.connect(self.flip_sprite_vertical); t_menu.addAction(fv)
            t_menu.addSeparator()
            rr=QAction("Rotar 90° D",self); rr.triggered.connect(self.rotate_sprite_right_90); t_menu.addAction(rr)
            rl=QAction("Rotar 90° I",self); rl.triggered.connect(self.rotate_sprite_left_90); t_menu.addAction(rl)
            menu.addMenu(t_menu); menu.addSeparator()
            dup_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView)
            dup=QAction(dup_icon, "Duplicar",self); dup.triggered.connect(self.duplicate_sprite); menu.addAction(dup); menu.addSeparator()
        save=QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),"Guardar Sel...",self); save.setEnabled(n_sel > 0); save.triggered.connect(self.save_selected_sprites); menu.addAction(save)
        delete=QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon),"Eliminar Sel...",self); delete.setEnabled(n_sel > 0); delete.triggered.connect(self.delete_selected_sprites); menu.addAction(delete)
        menu.exec(self.preview_list_widget.mapToGlobal(position))

    @pyqtSlot()
    def start_apply_effects(self):
        self.update_log_output("--- Aplicar Efectos ---");
        if self.is_currently_busy(): return
        if not self.original_extracted_sprites: self.show_message_box("Info", "No sprites originales.", "info"); return
        p={'pixelate': self.pixelate_checkbox.isChecked(), 'pixel_size': self.pixelate_size_spinbox.value(), 'quantize': self.quantize_checkbox.isChecked(), 'quantize_colors': self.quantize_colors_spinbox.value(), 'quantize_method': self.quantize_method_combobox.currentText()};
        if not p['pixelate'] and not p['quantize']: self.show_message_box("Info","Selecciona efectos.","info"); return;
        if self.effects_worker: self.effects_worker.quit(); self.effects_worker.wait(100); self.effects_worker.deleteLater(); self.effects_worker = None
        self.set_buttons_enabled_status(False); self.update_log_output("Worker efectos..."); self.progress_bar.setRange(0,100); self.progress_bar.setValue(0); self.progress_bar.setVisible(True);
        try:
            self.effects_worker = EffectsWorker(list(self.original_extracted_sprites), p);
            self.effects_worker.progress.connect(self.update_log_output,Qt.ConnectionType.QueuedConnection); self.effects_worker.effect_applied.connect(self.update_processed_sprite_and_preview,Qt.ConnectionType.QueuedConnection); self.effects_worker.error.connect(self.handle_effects_worker_error,Qt.ConnectionType.QueuedConnection); self.effects_worker.finished.connect(self.handle_effects_worker_finished,Qt.ConnectionType.QueuedConnection); self.effects_worker.progress_update.connect(self.update_progress_bar_value,Qt.ConnectionType.QueuedConnection);
            self.effects_worker.finished.connect(self._cleanup_effects_qthread_object);
            self.effects_worker.start();
        except Exception as e: self.update_log_output(f"Err start efectos: {e}"); self.show_message_box("Error",f"Err efectos: {e}","error"); self._clear_and_nullify_effects_worker(); self.set_buttons_enabled_status(True); self.progress_bar.setVisible(False);

    @pyqtSlot()
    def _cleanup_effects_qthread_object(self):
        sender_worker = self.sender();
        if sender_worker and isinstance(sender_worker, EffectsWorker): sender_worker.deleteLater();
        if self.effects_worker == sender_worker: self.effects_worker = None

    @pyqtSlot(int, object)
    def update_processed_sprite_and_preview(self, sprite_index, processed_pil_image):
        if not isinstance(processed_pil_image, Image.Image): self.update_log_output(f"Err: Update sprite {sprite_index} no válido."); return
        if 0 <= sprite_index < len(self.processed_sprites) and 0 <= sprite_index < self.preview_list_widget.count():
            try:
                self.processed_sprites[sprite_index] = processed_pil_image.copy()
                item = self.preview_list_widget.item(sprite_index)
                if item:
                    tq = self.preview_list_widget.iconSize(); ts = (tq.width(), tq.height()); ts = ts if ts[0]>0 and ts[1]>0 else (64,64)
                    canvas = processed_pil_image; final_thumb = canvas
                    if final_thumb.mode != 'RGBA': temp = final_thumb.copy(); final_thumb = temp.convert('RGB').convert('RGBA') if temp.mode == 'P' else temp.convert('RGBA')
                    thumb = final_thumb.resize(ts, Image.Resampling.LANCZOS); qimg = ImageQt.ImageQt(thumb); pixmap = QPixmap.fromImage(qimg)
                    if not pixmap.isNull(): item.setIcon(QIcon(pixmap))
                    else: self.update_log_output(f"Warn: Falló QPixmap thumb {sprite_index + 1} post-efecto.");
            except Exception as e: self.update_log_output(f"Err update sprite/thumb {sprite_index + 1}: {e}")
        else: self.update_log_output(f"Adv: Índice {sprite_index + 1} fuera rango update.");

    @pyqtSlot(str)
    def handle_effects_worker_error(self, error_message):
        self.update_log_output(f"ERROR EFECTOS:\n{error_message}"); self.show_message_box("Error Efectos", error_message, "error");
        if self.effects_worker: self.effects_worker.deleteLater(); self._clear_and_nullify_effects_worker()
        self.progress_bar.setVisible(False); self.set_buttons_enabled_status(True)

    @pyqtSlot(str)
    def handle_effects_worker_finished(self, message_from_worker):
        self.update_log_output(f"--- {message_from_worker} ---"); self.show_message_box("Efectos OK", message_from_worker, "info");
        if self.effects_worker: self.effects_worker.deleteLater(); self._clear_and_nullify_effects_worker()
        self.progress_bar.setVisible(False); self.set_buttons_enabled_status(True);

    @pyqtSlot()
    def revert_to_originals(self):
        self.update_log_output("--- Revertir ---");
        if self.is_currently_busy(): return
        if not self.original_extracted_sprites: self.show_message_box("Info", "No originales.", "info"); return
        if len(self.original_extracted_sprites)!=self.preview_list_widget.count() or len(self.original_extracted_sprites)!=len(self.processed_sprites): self.show_message_box("Error","Inconsistencia.","error"); return;
        try:
            reply=QMessageBox.question(self,"Confirmar","Descartar efectos?",QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No);
            if reply==QMessageBox.StandardButton.No: return;
            self.update_log_output("Revirtiendo..."); self.set_buttons_enabled_status(False);
            self.processed_sprites=[s.copy() for s in self.original_extracted_sprites]; total=len(self.processed_sprites);
            self.progress_bar.setRange(0,total); self.progress_bar.setValue(0); self.progress_bar.setVisible(True);
            for idx in range(total):
                self.update_processed_sprite_and_preview(idx,self.processed_sprites[idx]); self.progress_bar.setValue(idx+1);
                if idx%20==0: QApplication.processEvents();
            QApplication.processEvents(); self.progress_bar.setVisible(False); self.update_log_output("Revertido OK."); self.show_message_box("Revertido","Efectos descartados.","info");
        except Exception as e: self.update_log_output(f"¡ERR revert!\n{e}"); self.show_message_box("Error",f"Err revert: {e}","error"); self.progress_bar.setVisible(False);
        finally: self.set_buttons_enabled_status(True);

    # --- MÉTODO START_REGENERATION CORREGIDO ---
    @pyqtSlot()
    def start_regeneration(self):
        self.update_log_output("--- Iniciando creación de Spritesheet ---")
        if self.is_currently_busy(): return

        sprites_to_use = self.processed_sprites
        if not sprites_to_use: self.show_message_box("Error", "No hay sprites.", "error"); return

        resize_enabled = self.resize_output_checkbox.isChecked()
        final_sprites_for_sheet = []
        target_width = 0; target_height = 0; sprite_width = 0; sprite_height = 0

        if resize_enabled:
            target_width = self.resize_output_width_spinbox.value(); target_height = self.resize_output_height_spinbox.value()
            resize_filter_name = self.resize_output_resampling_combobox.currentText(); resize_filter = self._get_resampling_filter(resize_filter_name)
            self.update_log_output(f"Redimensionando sprites para sheet a {target_width}x{target_height} ({resize_filter_name})...")
            if target_width <= 0 or target_height <= 0: self.show_message_box("Error", "Dimensiones resize inválidas.", "error"); return
            temp_resized = []; success = True
            for idx, sprite_img in enumerate(sprites_to_use):
                 try: temp_resized.append(sprite_img.resize((target_width, target_height), resample=resize_filter))
                 except Exception as e: self.update_log_output(f"Err resize sprite {idx} para sheet: {e}. Abortando."); self.show_message_box("Error", f"Error redim sprite {idx+1}.", "error"); success = False; break
            if not success: self.set_buttons_enabled_status(True); return
            final_sprites_for_sheet = temp_resized
            if not final_sprites_for_sheet: self.show_message_box("Error", "No sprites válidos post-resize.", "error"); self.set_buttons_enabled_status(True); return
            self.update_log_output(f"{len(final_sprites_for_sheet)} sprites redimensionados."); sprite_width, sprite_height = target_width, target_height
        else:
            final_sprites_for_sheet = list(sprites_to_use)
            try: sprite_width, sprite_height = final_sprites_for_sheet[0].size;
            except IndexError: self.show_message_box("Error", "Lista de sprites vacía.", "error"); return
            if sprite_width <= 0 or sprite_height <= 0: self.show_message_box("Error", "Sprites con tamaño inválido.", "error"); return

        output_directory_path = self.output_dir_lineedit.text(); is_output_dir_valid = False
        if output_directory_path:
            if os.path.isdir(output_directory_path): is_output_dir_valid = True
            elif not os.path.exists(output_directory_path):
                try: os.makedirs(output_directory_path, exist_ok=True); is_output_dir_valid = True
                except OSError as e: self.update_log_output(f"No crear dir: {output_directory_path}. Err: {e}")
            else: self.update_log_output(f"Ruta no es dir: {output_directory_path}")
        if not is_output_dir_valid:
             output_directory_path = self.output_dir
             try: os.makedirs(output_directory_path, exist_ok=True); self.output_dir_lineedit.setText(output_directory_path); self.update_log_output(f"Usando dir default: {output_directory_path}")
             except OSError as e: self.show_message_box("Error", f"No crear dir:\n{e}","error"); return

        if self.regeneration_worker: self.regeneration_worker.quit(); self.regeneration_worker.wait(100); self.regeneration_worker.deleteLater(); self.regeneration_worker = None
        self.set_buttons_enabled_status(False); self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0); self.progress_bar.setVisible(True)

        try:
            layout_mode = 'Columns' if self.regen_layout_cols_radio.isChecked() else 'Rows'
            cols_value = self.spritesheet_columns_spinbox.value(); rows_value = self.spritesheet_rows_spinbox.value(); spacing_pixels = self.spritesheet_spacing_spinbox.value()

            base_filename_prefix = "spritesheet"
            if self.input_paths:
                 # --- CORRECCIÓN DE SINTAXIS APLICADA ---
                 try:
                     base_filename_prefix = os.path.splitext(os.path.basename(self.input_paths[0]))[0]
                 except (IndexError, TypeError):
                      base_filename_prefix = "sprite_lote"
                 # --- FIN CORRECCIÓN ---
            elif final_sprites_for_sheet:
                base_filename_prefix = "sprites_cargados"

            size_suffix = f"{sprite_width}x{sprite_height}px" + ("_resized" if resize_enabled else "")
            layout_suffix = f"{layout_mode[:3].lower()}{cols_value if layout_mode == 'Columns' else rows_value}"
            sheet_filename = f"{base_filename_prefix}_sheet_{size_suffix}_{layout_suffix}_{spacing_pixels}sp.png"
            full_path = os.path.join(output_directory_path, sheet_filename)
            self.update_log_output(f"  Sheet: {sheet_filename}")

            self.regeneration_worker = SpritesheetWorker(final_sprites_for_sheet, layout_mode, cols_value, rows_value, spacing_pixels, full_path)
            self.regeneration_worker.progress.connect(self.update_log_output, Qt.ConnectionType.QueuedConnection); self.regeneration_worker.error.connect(self.handle_regeneration_worker_error, Qt.ConnectionType.QueuedConnection); self.regeneration_worker.finished.connect(self.handle_regeneration_worker_finished, Qt.ConnectionType.QueuedConnection); self.regeneration_worker.progress_update.connect(self.update_progress_bar_value, Qt.ConnectionType.QueuedConnection); self.regeneration_worker.finished.connect(self._cleanup_regeneration_qthread_object)
            self.regeneration_worker.start(); self.update_log_output("  Worker regen iniciado.")
        except Exception as e: self.update_log_output(f"¡ERROR iniciar regen!\n{e}\n{traceback.format_exc()}"); self.show_message_box("Error", f"Error: {e}", "error"); self._clear_and_nullify_regeneration_worker(); self.set_buttons_enabled_status(True); self.progress_bar.setVisible(False)

    @pyqtSlot()
    def _cleanup_regeneration_qthread_object(self):
        sender_worker = self.sender()
        if sender_worker and isinstance(sender_worker, SpritesheetWorker):
            sender_worker.deleteLater()
            if self.regeneration_worker == sender_worker: self.regeneration_worker = None

    @pyqtSlot(str)
    def handle_regeneration_worker_error(self, error_message):
        self.update_log_output(f"ERR REGEN:\n{error_message}"); self.show_message_box("Error Regen",error_message,"error");
        if self.regeneration_worker: self.regeneration_worker.deleteLater(); self._clear_and_nullify_regeneration_worker()
        self.progress_bar.setVisible(False); self.set_buttons_enabled_status(True);

    @pyqtSlot(str)
    def handle_regeneration_worker_finished(self, message_from_worker):
        self.update_log_output(f"\n---\n{message_from_worker}\n---"); self.show_message_box("Regen OK",message_from_worker);
        if self.regeneration_worker: self.regeneration_worker.deleteLater(); self._clear_and_nullify_regeneration_worker()
        self.progress_bar.setVisible(False); self.set_buttons_enabled_status(True);

    def set_buttons_enabled_status(self, enabled_flag):
        is_app_busy = self.is_currently_busy(); actual_enabled_state = enabled_flag and not is_app_busy
        self.extract_button.setEnabled(actual_enabled_state); self.input_button.setEnabled(actual_enabled_state); self.output_dir_button.setEnabled(actual_enabled_state); self.load_sprites_button.setEnabled(actual_enabled_state)
        has_sprites = bool(self.processed_sprites)
        # Grupo principal de salida
        self.output_options_groupbox.setEnabled(actual_enabled_state and has_sprites)
        # Controles dentro del grupo de salida (su habilitación depende del grupo padre)
        parent_is_enabled = self.output_options_groupbox.isEnabled()
        self.generate_spritesheet_button.setEnabled(parent_is_enabled)
        self.regen_layout_cols_radio.setEnabled(parent_is_enabled)
        self.regen_layout_rows_radio.setEnabled(parent_is_enabled)
        self._toggle_regen_spinboxes(self.regen_layout_cols_radio.isChecked()) # Actualiza spinboxes cols/rows
        self.spritesheet_spacing_spinbox.setEnabled(parent_is_enabled)
        self.resize_output_checkbox.setEnabled(parent_is_enabled)
        is_resize = self.resize_output_checkbox.isChecked(); can_enable_resize = parent_is_enabled and is_resize
        self.resize_output_width_spinbox.setEnabled(can_enable_resize); self.resize_output_height_spinbox.setEnabled(can_enable_resize); self.resize_output_resampling_combobox.setEnabled(can_enable_resize)
        # Otros controles
        self.apply_effects_button.setEnabled(actual_enabled_state and has_sprites); self.revert_effects_button.setEnabled(actual_enabled_state and has_sprites)
        self.update_delete_button_state()
        self.engine_combobox.setEnabled(actual_enabled_state); self.min_area_spinbox.setEnabled(actual_enabled_state); self.resampling_combobox.setEnabled(actual_enabled_state); self.custom_size_checkbox.setEnabled(actual_enabled_state)
        is_custom = self.custom_size_checkbox.isChecked(); self.custom_size_spinbox.setEnabled(actual_enabled_state and is_custom);
        for rb in self.size_radiobuttons.values(): rb.setEnabled(actual_enabled_state and not is_custom)
        self.pixelate_checkbox.setEnabled(actual_enabled_state and has_sprites); self.quantize_checkbox.setEnabled(actual_enabled_state and has_sprites)
        self.pixelate_size_spinbox.setEnabled(actual_enabled_state and has_sprites and self.pixelate_checkbox.isChecked())
        self.quantize_colors_spinbox.setEnabled(actual_enabled_state and has_sprites and self.quantize_checkbox.isChecked())
        self.quantize_method_combobox.setEnabled(actual_enabled_state and has_sprites and self.quantize_checkbox.isChecked())

    def dragEnterEvent(self, event:QDragEnterEvent):
        if self.is_currently_busy(): event.ignore(); return
        mime = event.mimeData();
        if mime.hasUrls(): urls=mime.urls(); ext=('.png','.jpg','.jpeg','.bmp','.webp');
        if urls and any(u.isLocalFile() and u.toLocalFile().lower().endswith(ext) for u in urls): event.acceptProposedAction(); return;
        event.ignore();

    def dropEvent(self, event: QDropEvent):
        if self.is_currently_busy(): event.ignore(); return
        urls = event.mimeData().urls()
        if urls:
            valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp'); dropped = []; ignored = []
            for u in urls:
                fp = None;
                if u.isLocalFile(): fp = u.toLocalFile()
                if fp and os.path.isfile(fp) and fp.lower().endswith(valid_ext): dropped.append(fp)
                elif fp: ignored.append(os.path.basename(fp))
                else: ignored.append("URL no local")
            if ignored: self.update_log_output(f"Ignorados (drop): {', '.join(ignored)}")
            if dropped: self.update_log_output(f"Archivos soltados: {len(dropped)}"); event.acceptProposedAction(); self.process_selected_input_files(dropped)
            else: event.ignore()
        else: event.ignore()

    def closeEvent(self, event):
        self.update_log_output("--- Cerrando ---"); print("Cerrando..."); active = []
        if self.extraction_worker and self.extraction_worker.isRunning(): active.append({"w":self.extraction_worker,"t":self.extraction_worker,"n":"Extr"})
        if self.regeneration_worker and self.regeneration_worker.isRunning(): active.append({"w":self.regeneration_worker,"t":self.regeneration_worker,"n":"Regen"})
        if self.effects_worker and self.effects_worker.isRunning(): active.append({"w":self.effects_worker,"t":self.effects_worker,"n":"Efectos"})
        if active:
            print(f"Esperando {len(active)} hilos..."); QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor); QApplication.processEvents();
            for info in active:
                w,t,n=info["w"],info["t"],info["n"]; print(f"Deteniendo {n}..."); self.update_log_output(f"Deteniendo {n}...");
                # --- CORRECCIÓN DE SINTAXIS APLICADA ---
                if w and hasattr(w, 'stop'):
                    try:
                        w.stop()
                    except Exception as e_stop:
                         print(f"Excepción al llamar a stop() en worker '{n}': {e_stop}")
                # --- FIN CORRECCIÓN ---
                if not t.wait(3000): print(f"Adv: Hilo {n} no terminó."); t.terminate(); t.wait(500);
                else: print(f"{n} detenido.");
                QApplication.processEvents();
            QApplication.restoreOverrideCursor();
        print("Cierre OK."); event.accept();


# --- Punto de Entrada Principal ---
if __name__ == '__main__':
    if getattr(sys, 'frozen', False): application_path = os.path.dirname(sys.executable)
    else:
        try: application_path = os.path.dirname(os.path.abspath(__file__))
        except NameError: application_path = os.getcwd()

    config_file_path = os.path.join(application_path, "config.json")
    config_data = load_config(config_file_path)

    application = QApplication(sys.argv)
    try:
        main_window = SpriteExtractorProApp(config=config_data)
        main_window.show()
    except Exception as init_exception:
        print(f"ERROR CRÍTICO INICIALIZACIÓN:\n{init_exception}\n{traceback.format_exc()}")
        try:
            error_box = QMessageBox(); error_box.setIcon(QMessageBox.Icon.Critical); error_box.setWindowTitle("Error Crítico")
            error_box.setText(f"Error fatal al iniciar:\n{init_exception}"); error_box.setDetailedText(traceback.format_exc()); error_box.exec()
        except Exception as e_msgbox: print(f"No se pudo mostrar QMessageBox: {e_msgbox}")
        sys.exit(1)
    sys.exit(application.exec())

# --- END OF FILE appV17b_final_syntax_fix_v5.py ---