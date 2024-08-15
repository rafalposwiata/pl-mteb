import React, {ReactNode} from "react";
import {EvalData, FilterDef} from "../model";
import {TableView} from "./table";
import {Tooltip} from "react-tooltip";
import {copyTextToClipboard} from "../clipboard";

interface AppViewState {
    data: EvalData | null;
    metric: string | null;
    filters: {[key: string]: string};
    filterSet: Set<string>;
}

export class AppView extends React.Component<any, AppViewState> {
    constructor(props: any) {
        super(props);
        this.state = {data: null, metric: "main_score", filters: {}, filterSet: new Set()};
    }

    componentDidMount() {
        let url = `${window.location.protocol}//${window.location.host}`;
        url += "/data.json";
        fetch(url)
            .then(val => val.json())
            .then(val => this.setState({data: val, filterSet: this.defaultFilters(val)}));
    }

    private defaultFilters(data: EvalData) {
        const filters = data?.filters || [];
        const filterSet = new Set<string>();
        for (const filter of filters) {
            const filterOptions = filter.options;
            if (filterOptions.length > 0) {
                filterSet.add(filterOptions[0].tag)
            }
        }
        return filterSet;
    }

    private renderMetrics(): ReactNode {
        const metrics = this.state.data?.metrics || [];
        const selected = this.state.metric || metrics[0].id;
        const options = [];
        for (const metric of metrics) {
            options.push(
                <input type="radio" id={metric.id} name="metric" key={metric.id}
                       value={metric.id} checked={metric.id === selected}
                       onChange={(event) => this.selectMetric(event)}>
                </input>
            );
            options.push(<label htmlFor={metric.id} key={"label " + metric.id}>{metric.name || metric.id}</label>)
        }
        return (
            <div className="filter-block" key="metric">
                <strong>Evaluation metric:</strong>
                {options}
            </div>
        )
    }

    private renderFilters(): ReactNode[] {
        const filters = this.state.data?.filters || [];
        return filters.map(val => this.renderFiltersBlock(val));
    }

    private renderFiltersBlock(spec: FilterDef): ReactNode {
        const filterOptions = spec.options;
        const selected = this.state.filters[spec.name] || filterOptions[0].tag;
        const options = [];
        for (const opt of filterOptions) {
            options.push(
                <input type="radio" id={opt.tag} name={spec.name} key={opt.tag}
                       value={opt.tag} checked={opt.tag === selected}
                       onChange={(event) => this.selectFilter(spec.name, event)}>
                </input>
            );
            options.push(<label htmlFor={opt.tag} key={"label " + opt.tag}>{opt.name || opt.tag}</label>)
        }
        return <div className="filter-block" key={spec.name}><strong>{spec.name}:</strong>{options}</div>;
    }

    private selectMetric(event: any) {
        this.setState({...this.state, metric: event.target.value});
    }

    private selectFilter(filter: string, event: any) {
        let filters = {...this.state.filters};
        filters[filter] = event.target.value === "none" ? null : event.target.value || null;
        const filterSet = new Set<string>(Object.values(filters).filter(val => val != null));
        this.setState({...this.state, filters: filters, filterSet: filterSet});
    }

    renderHelp() {
        return (
            <p className="description">
                Above each task group there is a panel which contains the following actions:
                <span className="circled-icon">+</span> expands the group into individual tasks,
                <span className="circled-icon">?</span> displays a tooltip providing additional information about the group,
                <span className="circled-icon">🔗</span> opens a link to the paper or the official website,
                <span className="circled-icon">✕</span> closes the group and removes it from the evaluation.
            </p>
        )
    }

    renderPaperInfo() {
        let el = [];
        if (this.state.data?.paper) {
            el.push(<a key="paper" href={this.state.data.paper} target="_blank">🗋 Paper</a>);
        }
        if (this.state.data?.citation) {
            el.push(
                <a key="cite" onClick={() => copyTextToClipboard(this.state.data?.citation)}
                   data-tooltip-id="table-tooltip" data-tooltip-content={"Citation copied to clipboard!"}>
                    🗍 Cite
                </a>
            );
        }
        return el.length == 0 ? null : <span className={"paper-info"}>{el}</span>;
    }

    render() {
        if (this.state.data == null) return null;
        const metric: string = this.state.metric || this.state.data.metrics[0].id;
        const desc = this.state.data.description ? <p className="description" dangerouslySetInnerHTML={{__html: this.state.data.description}}></p> : null;
        const help = this.state.data.options?.showHelp ? this.renderHelp() : null;
        let footer = null;
        if (this.state.data.options?.showFooter) {
             footer = <div className="footer">Author: <a target="_blank" href="https://github.com/sdadas" rel="noreferrer">Sławomir Dadas</a></div>;
        }
        return (
            <div>
                <h1 style={{"maxWidth": "1200px"}}>
                    {this.state.data.title}
                    {this.renderPaperInfo()}
                </h1>
                {desc}{help}<br/>
                {/*{this.renderMetrics()}*/}
                {this.renderFilters()}
                <TableView data={this.state.data} mainMetric={metric} tags={this.state.filterSet} />
                <Tooltip id="table-tooltip" openOnClick={true} variant="light" />
                {footer}
            </div>
        )
    }
}