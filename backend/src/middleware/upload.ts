import multer from "multer";

// Files are held in memory; routes persist them to temp dirs as needed.
export const upload = multer({ storage: multer.memoryStorage() });
