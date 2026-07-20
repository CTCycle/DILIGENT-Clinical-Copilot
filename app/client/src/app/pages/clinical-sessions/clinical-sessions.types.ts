export type ClinicalSessionSection = 'preview' | 'editor' | 'metadata' | 'revision' | 'timeline';
export type ClinicalSessionDateFilterMode = 'any' | 'after' | 'before' | 'exact';
export type EditorViewMode = 'source' | 'rendered';

export type EditorCommandName =
  | 'formatBlock'
  | 'undo'
  | 'redo'
  | 'bold'
  | 'italic'
  | 'strikeThrough'
  | 'insertUnorderedList'
  | 'insertOrderedList'
  | 'createLink';

export type EditorCommandEvent = {
  command: EditorCommandName;
  value?: string;
};
