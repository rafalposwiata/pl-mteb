
export interface EvalData {
    title: string;
    description?: string;
    paper?: string;
    citation?: string;
    options?: EvalDataOptions;
    metrics: ObjectDef[];
    taskGroups: ObjectDef[];
    tasks: ObjectDef[];
    models: ObjectDef[];
    results: ModelResults[];
    filters?: FilterDef[];
}

export interface EvalDataOptions {
    closeable?: boolean;
    expandable?: boolean;
    showHelp?: boolean;
    showFooter?: boolean;
}

export interface ObjectDef {
    id: string;
    name?: string;
    description?: string;
    url?: string;
    groupId?: string;
    warning?: string;
    tags?: string[];
}

export interface FilterDef {
    name: string;
    options: FilterOption[];
}

export interface FilterOption {
    id: string;
    name: string;
    tag: string;
}

export interface ModelResults extends ObjectDef {
    results: any;
}

export enum CellType {
    EMPTY = "EMPTY",
    HEADER_GROUP = "HEADER_GROUP",
    SORTABLE_HEADER = "SORTABLE_HEADER",
    HEADER = "HEADER",
    VALUE = "VALUE",
    ROWID = "ROWID",
    ROWID_HEADER = "ROWID_HEADER"
}

export interface Cell {
    value: string;
    type: CellType;
    colspan?: number;
    classes?: string;
    columnId?: string;
    url?: string;
    description?: string;
    warning?: string;
}

export class Cells {

    public static empty(classes: string=""): Cell {
        return {value: "", type: CellType.EMPTY, classes: classes};
    }
}