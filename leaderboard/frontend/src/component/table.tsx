// noinspection PointlessBooleanExpressionJS
import React, {ReactNode} from "react";
import {Cell, Cells, CellType, EvalData, ModelResults, ObjectDef} from "../model";
import {InfoIcon} from "./info";
import * as XLSX from 'xlsx';
import {BookType} from "xlsx";
const DoubleScrollbar = require("react-double-scrollbar");

interface TableProps {
    data: EvalData;
    mainMetric: string;
    tags: Set<string>;
}

interface TableState {
    sortId: string;
    filter: string;
    expanded: {[key: string]: boolean};
    closed: {[key: string]: boolean};
}

export class TableView extends React.Component<TableProps, TableState> {

    private ranking: RankingHelper;

    constructor(props: Readonly<TableProps> | TableProps) {
        super(props);
        const expanded: {[key: string]: boolean} = {};
        (props.data?.taskGroups || []).forEach(val => expanded[val.id] = false);
        const closed: {[key: string]: boolean} = {};
        this.state = {sortId: `${RankingHelper.GLOBAL}_average`, expanded: expanded, closed: closed, filter: ""}
        this.ranking = new RankingHelper(props.data, props.mainMetric, props.tags, this.state);
    }


    shouldComponentUpdate(nextProps: Readonly<TableProps>, nextState: Readonly<TableState>, nextContext: any): boolean {
        if (this.state.filter !== nextState.filter) {
            /* DO NOTHING */
        } else {
            this.ranking = new RankingHelper(nextProps.data, nextProps.mainMetric, nextProps.tags, nextState);
        }
        return true;
    }

    componentDidUpdate(prevProps: Readonly<TableProps>, prevState: Readonly<any>, snapshot?: any) {
        this.ranking = new RankingHelper(this.props.data, this.props.mainMetric, this.props.tags, this.state);
    }

    private renderTable(): ReactNode {
        const topHeader = this.renderRow(this.ranking.topHeader(), true, -2, "topHeader");
        const mainHeader = this.renderRow(this.ranking.mainHeader(), true, -1, "mainHeader");
        const header = <thead>{topHeader}{mainHeader}</thead>;
        const rows = this.ranking.rows();
        const rowElements: ReactNode[] = [];
        for (let i = 0; i < rows.length; i++) {
            const row = rows[i];
            const rowKey = row[0].value;
            rowElements.push(this.renderRow(row, false, i, rowKey));
        }
        const body = <tbody>{rowElements}</tbody>
        return <table>{header}{body}</table>;
    }

    private exportTable(event: any, format: BookType) {
        event.preventDefault();
        let res = [];
        res.push(this.exportRow(this.ranking.topHeader()));
        res.push(this.exportRow(this.ranking.mainHeader()));
        const rows = this.ranking.rows();
        for (const row of rows) {
            res.push(this.exportRow(row));
        }
        const book = XLSX.utils.book_new();
        const sheet = XLSX.utils.aoa_to_sheet(res);
        XLSX.utils.book_append_sheet(book, sheet, "export");
        XLSX.writeFile(book, `export.${format}`, {bookType: format})
    }

    private exportRow(cells: Cell[]) {
        return cells.map(val => this.exportCell(val)).flat();
    }

    private exportCell(cell: Cell) {
        const colspan = cell.colspan || 1;
        let value: string | number = cell.value.replace("\n", " ");
        if (!isNaN(parseFloat(value))) {
            value = parseFloat(value);
        }
        if (colspan <= 1) {
            return value;
        } else {
            let res = Array(colspan).fill("")
            res[0] = value;
            return res;
        }
    }

    private renderRow(cells: Cell[], header: boolean, rowIdx: number, rowKey: string): ReactNode {
        const row = cells.map((val, idx) => this.renderCell(val, header, rowIdx, idx));
        return <tr key={rowKey}>{row}</tr>
    }

    private renderCell(c: Cell, header: boolean, rowIdx: number, colIdx: number): ReactNode {
        let clazz = `cell-${c.type.toLowerCase()} ${c.classes ? c.classes : ''}`;
        const key = `${rowIdx}_${colIdx}_${c.value}`;
        if (header) {
            if (c.type === CellType.SORTABLE_HEADER) {
                return this.renderSortableColumn(c, key);
            } else if (c.type === CellType.HEADER_GROUP) {
                return this.renderHeaderGroupColumn(c, key);
            } else if (c.type === CellType.ROWID_HEADER) {
                return this.renderRowidHeader(c, key);
            } else {
                return <th colSpan={c.colspan || 1} className={clazz} key={key}>{c.value}</th>;
            }
        } else {
            if (c.type === CellType.ROWID) {
                return this.renderRowidCell(c, key);
            } else {
                return <td colSpan={c.colspan || 1} className={clazz} key={key}>{c.value}</td>;
            }
        }
    }

    private renderRowidHeader(c: Cell, key: string): ReactNode {
        let clazz = `cell-${c.type.toLowerCase()} ${c.classes ? c.classes : ''}`;
        const search = (
            <div className="rowid-filter">
                <span className="icon">🔍</span>
                <input type="text" placeholder="Filter models"
                   onChange={e => this.filter(e)} value={this.state.filter}>
                </input>
                <div className="export-panel">
                    <button onClick={e => this.exportTable(e, "xlsx")}>XLSX</button>
                    <button onClick={e => this.exportTable(e, "csv")}>CSV</button>
                    <button onClick={e => this.exportTable(e, "html")}>HTML</button>
                </div>
            </div>
        )
        return <td colSpan={c.colspan || 1} className={clazz} key={key}>{search}</td>
    }

    private renderRowidCell(c: Cell, key: string) {
        let clazz = `cell-${c.type.toLowerCase()} ${c.classes ? c.classes : ''}`;
        let value: string | ReactNode = c.value;
        if (c.url) {
            value = <a target="_blank" rel="noreferrer" href={c.url}>{c.value}</a>
        }
        let desc = c.description ? <InfoIcon text={c.description} /> : null;
        let warn = c.warning ? <InfoIcon text={c.warning} warning={true} /> : null;
        return <td colSpan={c.colspan || 1} className={clazz} key={key}>{value}{desc}{warn}</td>;
    }

    private renderHeaderGroupColumn(c: Cell, key: string): ReactNode {
        let clazz = `cell-${c.type.toLowerCase()} ${c.classes ? c.classes : ''} `;
        const expanded = this.state.expanded[c.value];
        clazz += expanded ? "cell-expanded-true" : "cell-expanded-false";
        const url = c.url ? <a target="_blank" rel="noreferrer" href={c.url} className="circled-icon">🔗</a> : null;
        const tooltip = c.description ? <InfoIcon text={c.description} /> : null;
        const closeable = this.props.data.options?.closeable || false;
        const close = closeable ? <span className="circled-icon" onClick={() => this.closeGroup(c.columnId)}>✕</span> : null;
        const expandable = this.props.data.options?.expandable || true;
        let expander = null;
        if (expandable) {
            const expanderSymbol = expanded ? "-" : "+";
            const expanderClasses = "expander-symbol circled-icon";
            expander = <span className={expanderClasses} onClick={() => this.toggleExpansion(c.columnId)}>{expanderSymbol}</span>;
        }
        return (
            <th colSpan={c.colspan || 1} className={clazz} key={key}>
                <div className="header-actions">
                    {expander}{tooltip}{url}{close}
                </div>
                <span>{c.value}</span>
            </th>
        );
    }

    private renderSortableColumn(c: Cell, key: string): ReactNode {
        let clazz = `cell-${c.type.toLowerCase()} ${c.classes ? c.classes : ''} `;
        let sortIndicator = <div className="sort-symbol">&nbsp;</div>;
        if (this.state.sortId === c.columnId) {
            clazz += " cell-sorted";
            sortIndicator = <div className="sort-symbol">▼</div>
        }
        return (
            <th colSpan={c.colspan || 1} className={clazz} onClick={() => this.sort(c.columnId)} key={key}>
                <span>{c.value}</span>
                {sortIndicator}
            </th>
        );
    }

    private filter(event: any) {
        const text = event.target.value || "";
        const newState = {...this.state, filter: text};
        this.setState(newState);
        this.ranking.updateState(newState);
    }

    private sort(columnId?: string) {
        if (!columnId) return;
        const newState = {...this.state, sortId: columnId};
        this.ranking.sort(columnId);
        this.setState(newState);
    }

    private toggleExpansion(columnId?: string) {
        if (!columnId) return;
        let expanded = {...this.state.expanded};
        expanded[columnId] = !expanded[columnId];
        const newState = {...this.state, expanded: expanded};
        this.ranking.updateState(newState);
        this.setState(newState);
    }

    private closeGroup(columnId?: string) {
        if (!columnId) return;
        let closed = {...this.state.closed};
        closed[columnId] = !closed[columnId];
        const newState = {...this.state, closed: closed};
        this.ranking.updateState(newState);
        this.setState(newState);
    }

    render() {
        return <div className="table-wrapper"><DoubleScrollbar>{this.renderTable()}</DoubleScrollbar></div>;
    }
}

class RankingHelper {

    public static readonly GLOBAL = "__global__";
    public static readonly TASKS_WON = "__tasks_won__";

    private groups: RankingColumnGroup[];
    private models: RankingModel[];
    private taskRanks: {[key: string]: string[]};

    constructor(private data: EvalData, private mainMetric: string, private tags: Set<string>, private tableState: TableState) {
        this.groups = this.createColumnGroups(data);
        this.models = this.createModels(data);
        this.computeAverages(data);
        this.taskRanks = this.computeTaskRanks();
        this.computeWinners();
        this.updateState(tableState);
    }

    public updateState(state: TableState) {
        this.tableState = state;
        const groups = [];
        for (const group of this.groups) {
            if (!state.closed[group.id]) {
                groups.push(group);
            }
        }
        if (groups.length !== this.groups.length) {
            this.groups = groups;
            this.computeAverages(this.data);
            this.taskRanks = this.computeTaskRanks();
            this.computeWinners();
        }
        if (state.sortId) {
            this.sort(state.sortId);
        }
        for (const group of this.groups) {
            group.expanded = state.expanded[group.id];
        }
    }

    public topHeader(): Cell[] {
        let res: Cell[] = [];
        res.push({value: "", type: CellType.ROWID_HEADER, classes: "cell-sticky"}); // models
        res.push(Cells.empty()); // tasks won
        res.push(Cells.empty("cell-border")); // global average
        this.groups.forEach(group => res.push(group.groupCell()));
        return res;
    }

    public mainHeader(): Cell[] {
        let res: Cell[] = [];
        res.push({value: "Model", type: CellType.HEADER, classes: "cell-sticky"});
        res.push({value: "Tasks won", type: CellType.SORTABLE_HEADER, columnId: RankingHelper.TASKS_WON});
        const avg = `Average\n(${this.tasks.length} tasks)`;
        res.push({value: avg, type: CellType.SORTABLE_HEADER, classes: "cell-border", columnId: this.avgKey()});
        for (const group of this.groups) {
            const avg = `Average\n(${group.tasks.length} tasks)`;
            res.push({value: avg, type: CellType.SORTABLE_HEADER, classes: "cell-border", columnId: this.avgKey(group.id)});
            if (group.expanded) {
                group.tasks.forEach(task => {
                    res.push({value: task.name || task.id, type: CellType.SORTABLE_HEADER, columnId: task.id});
                });
            }
        }
        return res;
    }

    public rows(): Cell[][] {
        let filter = null;
        if (this.tableState.filter) {
            filter = this.tableState.filter.toLowerCase().split(/[^\p{L}\d]/u).filter(val => val.length > 0);
        }
        let res: Cell[][] = [];
        for (const model of this.models) {
            if (this.isRowVisible(model, filter, this.tags)) {
                res.push(this.row(model));
            }
        }
        return res;
    }

    private isRowVisible(model: RankingModel, filter: string[] | null, tags: Set<string>) {
        if (tags.size > 0) {
            const modelTags = model.model.tags || [];
            let matches = false;
            for (const modelTag of modelTags) {
                if (tags.has(modelTag)) {
                    matches = true;
                    break;
                }
            }
            if (!matches) {
                return false;
            }
        }
        if (filter == null || filter.length === 0) return true;
        const name = (model.model.name || model.model.id).toLowerCase();
        for (const word of filter) {
            if (name.indexOf(word) < 0) {
                return false;
            }
        }
        return true;
    }

    public row(model: RankingModel): Cell[] {
        let res: Cell[] = [];
        res.push(model.rowidCell());
        res.push(model.tasksWonCell());
        res.push(model.valueCell(this.avgKey(), this.mainMetric, true));
        for (const group of this.groups) {
            res.push(model.valueCell(this.avgKey(group.id), this.mainMetric, true));
            if (group.expanded) {
                group.tasks.forEach(task => {
                    res.push(model.valueCell(task.id, this.mainMetric));
                });
            }
        }
        return res;
    }

    private createColumnGroups(data: EvalData): RankingColumnGroup[] {
        const res: any = {};
        let list: RankingColumnGroup[] = [];
        (data.taskGroups || []).map(val => new RankingColumnGroup(val)).forEach(val => {
            if (!this.tableState.closed[val.id]) {
                res[val.id] = val;
                list.push(val);
            }
        });
        const globals = new RankingColumnGroup({id: RankingHelper.GLOBAL});
        res[globals.id] = globals;
        list.push(globals);
        (data.tasks || []).forEach(task => {
            if (task.groupId && res[task.groupId]) {
                res[task.groupId].add(task);
            } else if (task.groupId && this.tableState.closed[task.groupId]) {
                /* DO NOTHING */
            } else {
                res[RankingHelper.GLOBAL].add(task);
            }
        });
        list = list.filter(val => val.tasks.length > 0);
        return list;
    }

    private createModels(data: EvalData): RankingModel[] {
        const defs: any = {};
        (data.models || []).forEach(val => defs[val.id] = val);
        let res: RankingModel[] = [];
        (data.results || []).forEach(val => {
            const modelDef = defs[val.id];
            if (modelDef) {
                val = {...val, ...modelDef};
            }
            res.push(new RankingModel(val));
        });
        return res;
    }

    private computeAverages(data: EvalData) {
        const metrics: string[] = data.metrics.map(val => val.id);
        for (const group of this.groups) {
            const key = this.avgKey(group.id);
            const tasks = group.tasks.map(task => task.id);
            this.models.forEach(val => this.computeAveragesColumn(val, key, tasks, metrics));
        }
        const globalKey = this.avgKey();
        const allTasks = this.tasks.map(task => task.id);
        this.models.forEach(val => this.computeAveragesColumn(val, globalKey, allTasks, metrics));
    }

    private computeAveragesColumn(model: RankingModel, key: string, tasks: string[], metrics: string[]) {
        for (const metric of metrics) {
            let sum: number | null = 0.0;
            for (const task of tasks) {
                const taskResults = model.model.results[task] || {};
                const value = taskResults[metric];
                if (typeof value === 'undefined' || value == null) {
                    sum = null;
                    break;
                } else {
                    sum += value;
                }
            }
            if (sum != null) {
                sum = sum / tasks.length;
                let results = model.model.results[key] || {};
                results[metric] = sum;
                model.model.results[key] = results;
            }
        }
    }

    public sort(column: string) {
        this.models.sort((a, b) => this.compare(a, b, column));
    }

    private compare(m1: RankingModel, m2: RankingModel, column: string) {
        const res1 = m1.model.results[column] || {};
        const res2 = m2.model.results[column] || {};
        let val1 = res1[this.mainMetric] || null;
        let val2 = res2[this.mainMetric] || null;
        if (column === RankingHelper.TASKS_WON) {
            val1 = m1.tasksWon;
            val2 = m2.tasksWon;
        }
        if (val1 == null && val2 == null) {
            return m1.model.id.localeCompare(m2.model.id);
        } else if (val1 == null) {
            return 1;
        } else if (val2 == null) {
            return -1;
        } else {
            let out = val2 - val1;
            if (out === 0) {
                return m1.model.id.localeCompare(m2.model.id);
            } else {
                return out;
            }
        }

    }

    private computeTaskRanks() {
        let taskRanks: {[key: string]: string[]} = {};
        let ids = this.tasks.map(task => task.id);
        this.groups.forEach(group => ids.push(this.avgKey(group.id)));
        ids.push(this.avgKey());
        for (const id of ids) {
            const models = [...this.models].filter(row => this.isRowVisible(row, null, this.tags));
            models.sort((a, b) => this.compare(a, b, id));
            taskRanks[id] = models
                .filter(val => typeof (val.model.results[id] || {})[this.mainMetric] === "number")
                .map(val => val.model.id);
        }
        return taskRanks;
    }

    private computeWinners() {
        const taskIds = this.tasks.map(val => val.id);
        for (const model of this.models) {
            model.computeWinners(this.taskRanks, taskIds);
        }
        const models = [...this.models];
        models.sort((a, b) => this.compare(a, b, RankingHelper.TASKS_WON));
        this.taskRanks[RankingHelper.TASKS_WON] = models.map(val => val.model.id);
        const otherColumnIds = this.groups.map(val => this.avgKey(val.id));
        otherColumnIds.push(this.avgKey());
        otherColumnIds.push(RankingHelper.TASKS_WON);
        for (const model of this.models) {
            model.computeAchievements(this.taskRanks, taskIds.concat(otherColumnIds));
        }
    }

    get tasks(): ObjectDef[] {
        let res: ObjectDef[] = [];
        this.groups.forEach(val => val.tasks.forEach(task => res.push(task)));
        return res;
    }

    private avgKey(groupId?: string) {
        return groupId ? `${groupId}_average` : `${RankingHelper.GLOBAL}_average`;
    }
}

class RankingColumnGroup {

    tasks: ObjectDef[];
    expanded: boolean = true;

    constructor(private taskGroup: ObjectDef) {
        this.tasks = [];
    }

    get id(): string {
        return this.taskGroup.id;
    }

    get colspan() {
        return this.expanded ? this.tasks.length + 1 : 1;
    }

    groupCell(): Cell {
        return {
            value: this.taskGroup.name || this.taskGroup.id,
            type: RankingHelper.GLOBAL === this.taskGroup.name ? CellType.EMPTY : CellType.HEADER_GROUP,
            colspan: this.colspan,
            classes: "cell-border",
            columnId: this.id,
            url: this.taskGroup.url,
            description: this.taskGroup.description
        }
    }

    add(task: ObjectDef) {
        this.tasks.push(task);
    }
}

class RankingModel {

    public tasksWon: number = 0;
    private achievements: any = {};

    constructor(public model: ModelResults) {
    }

    valueCell(column: string, metric: string, border: boolean=false): Cell {
        const results = this.model.results[column] || {};
        let value = results[metric];
        if (!value && value !== 0.0) {
            value = null;
        }
        const formatted = value != null ? value.toFixed(2).toString() : "";
        let classNames = border ? "cell-border " : " ";
        const medal = this.achievements[column];
        if (medal) {
            classNames += medal;
        }
        return {
            value: formatted,
            type: CellType.VALUE,
            classes: classNames
        }
    }

    rowidCell(): Cell {
        return {
            value: this.model.name || this.model.id,
            type: CellType.ROWID,
            classes: "cell-sticky",
            description: this.model.description,
            url: this.model.url,
            warning: this.model.warning
        }
    }

    tasksWonCell(): Cell {
        const medal = this.achievements[RankingHelper.TASKS_WON] || "";
        return {
            value: this.tasksWon.toFixed(0).toString(),
            type: CellType.VALUE,
            classes: medal
        }
    }

    computeWinners(taskRanks: {[key: string]: string[]}, taskIds: string[]) {
        let tasksWon = 0;
        for (const taskId of taskIds) {
            const ranking = taskRanks[taskId] || [];
            const place = ranking.indexOf(this.model.id);
            if (place === 0) {
                tasksWon += 1;
            }
        }
        this.tasksWon = tasksWon;
    }

    computeAchievements(taskRanks: {[key: string]: string[]}, columnIds: string[]) {
        let achievements: any = {};
        let medalsClasses: any = {0: "cell-gold", 1: "cell-silver", 2: "cell-bronze"};
        for (const taskId of columnIds) {
            const ranking = taskRanks[taskId] || [];
            const place = ranking.indexOf(this.model.id);
            const medal = medalsClasses[place];
            if (medal) {
                achievements[taskId] = medal;
            }
        }
        this.achievements = achievements;
    }
}